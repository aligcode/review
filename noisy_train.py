from noisy_dataset import CIFAR10, collate_fn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from loss import BootstrappedCrossEntropy, NTXentLoss
from mlp_model import ConvTwoLayerMLP
from torchmetrics.classification import F1Score
import os
import torch
import copy
from augs import ViewAugmentations
from diffusion_model import LabelDenoisingDiffusionModel
import argparse
import torch.distributed as dist
import random

class Trainer:
    
    def __init__(self, diffusion, diffusion_params, diffusion_model, num_diffusion_epochs, pretrain, num_pretrain_epochs, model, train_loader, val_loader, optimizers, loss, num_epochs, checkpoint_dir):
        
        self.diffusion = diffusion
        self.num_diffusion_epochs = num_diffusion_epochs
        self.pretrain = pretrain
        self.num_pretrain_epochs = num_pretrain_epochs
        self.augmentations = ViewAugmentations()

        if self.pretrain:
            self.nt_xent_loss = NTXentLoss()
            
        self.device_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.device_id)
        self.diffusion_model = diffusion_model.to(self.device_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizers['main_optimizer']
        self.loss = loss
        self.model = DDP(self.model, device_ids=[self.device_id], find_unused_parameters=True)
        
        if self.diffusion:
            assert self.pretrain # in order to use simclr embeddings
            self.img_emb_dim = diffusion_params['img_emb_dim']
            self.num_classes = diffusion_params['num_classes'] #10
            self.num_diffusion_steps = diffusion_params['num_diffusion_steps'] # 1000
            self.label_denoising_diffusion_model = DDP(self.diffusion_model, device_ids=[self.device_id], find_unused_parameters=True)
            self.label_denoising_diffusion_model.module.init_params(self.device_id)
            self.diffusion_optimizer = optimizers['diffusion_optimizer']
            
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
    
    def save_checkpoint(self, epoch, metrics, model_name):
        save_model = self.model if model_name == 'main_model' else self.label_denoising_diffusion_model
        checkpoint_info = {
            'model_state_dict': save_model.state_dict(),
            'epoch': epoch
        }
        if metrics != None:
            for k, v in metrics.items():
                checkpoint_info[k] = v
        save_name = 'best_model.pt' if model_name == 'main_model' else 'diffusion_model.pt'
        torch.save(checkpoint_info, os.path.join(self.checkpoint_dir, save_name))
        print(f"Saved {model_name} checkpoint. ")
        
    def load_checkpoint(self):
        ckpt_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        best_model = torch.load(ckpt_path)
        best_model_state_dict = best_model['model_state_dict']
        epoch, test_f1_score = -1, -1
        if "epoch" in best_model:
            epoch = best_model['epoch']
        if "test_f1_score" in best_model:
            test_f1_score = best_model['test_f1_score']
        self.model.load_state_dict(state_dict=best_model_state_dict)
        print(f"Loaded best model checkpoint from {ckpt_path} - Epoch {epoch} - F1-Score {test_f1_score}")
        
    def test(self): 
        metrics = {
            'test_loss': 0,
            'test_f1_score': 0
        }
        f1_metric = F1Score(task='multiclass', num_classes=10).to(self.device_id)
        
        self.model.eval()
        with torch.no_grad():
            for (x, y) in self.val_loader:
                x = self.no_augment(x)
                x, y = x.to(self.device_id), y.to(self.device_id)
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                preds = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
                f1_score = f1_metric(preds, y)
                metrics['test_loss'] += loss.detach().cpu().numpy()
                metrics['test_f1_score'] += f1_score
        
        if self.device_id == 0:    
            metrics['test_loss'] =  metrics['test_loss']/len(val_loader)
            metrics['test_f1_score'] =  metrics['test_f1_score']/len(val_loader)
            print(f"Test loss: {metrics['test_loss']}") 
            print(f"Test F1-Score: {metrics['test_f1_score']}")

        return metrics
    
    def augment(self, x):
        # list of PIL images
        # return tensor [batch, 3, H, W]
        augmented_tensors = []
        for pil_img in x:
            augmented_tensors.append(self.augmentations.apply_augmentation(pil_img))
        return torch.stack(augmented_tensors, dim=0)
    
    def no_augment(self, x):
        # list of PIL images
        normalized_tensors = []
        for pil_img in x:
            normalized_tensors.append(self.augmentations.normalize_tensorize(pil_img))
        
        normalized_tensors = torch.stack(normalized_tensors, dim=0)
        return normalized_tensors
        
    def close_pretrain(self):
        self.model.module.pretrain = False
        print("Pretraining turned off.")
    
    def run_denoising_model_train(self):
        
        for epoch in range(self.num_diffusion_epochs):
            train_loss = 0
            self.label_denoising_diffusion_model.train()
            self.model.eval()
            counts = 0
            for (x, y) in self.train_loader:
                self.diffusion_optimizer.zero_grad()
                x = self.no_augment(x)
                x, y = x.to(self.device_id), y.to(self.device_id)
                with torch.no_grad():
                    # get the embedding from simclr
                    x_embed_simclr = self.model(x, encode_only=True) # [batch, 128]
                    
                # randomly sample ts
                n = x_embed_simclr.shape[0]
                t = torch.randint(low=0, high=self.num_diffusion_steps, size=(n//2+1, )).to(self.device_id)
                ts = torch.cat([t, self.num_diffusion_steps-t-1], dim=0)[:n]
                noise_pred, noise_gt = self.label_denoising_diffusion_model.module.forward_t(x_embed=x_embed_simclr, x0=y, t=ts)
                loss = self.label_denoising_diffusion_model.module.get_diffusion_loss(noise_pred, noise_gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.label_denoising_diffusion_model.parameters(), 1.0)
                self.diffusion_optimizer.step()
                train_loss += loss.detach().cpu().numpy()
                counts += 1
                if self.device_id == 0 and counts % 200 == 0:
                    print(f"Diffusion loss idx {counts}: ", train_loss/(counts + 1))

            if self.device_id == 0:
                print(f"[DIFFUSION] Epoch {epoch} training loss:  {train_loss / len(train_loader)}")
                self.save_checkpoint(epoch, None, model_name='diffusion_model')
                
    def run_pretrain(self):
        
        for epoch in range(self.num_pretrain_epochs):
            train_loss = 0
            self.model.train()
            counts = 0 
            for (x, _) in self.train_loader:
                self.optimizer.zero_grad()
                x_view1, x_view2 = self.augment(x), self.augment(x)
                x_view1, x_view2 = x_view1.to(self.device_id), x_view2.to(self.device_id)
                p1, p2 = self.model(x_view1), self.model(x_view2)
                loss = self.nt_xent_loss(p1, p2)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.detach().cpu().numpy()
                if self.device_id == 0 and counts % 200 == 0:
                    print(f"Pre-Training loss idx {counts}: ", train_loss/(counts + 1))
                
                counts += 1
            
            if self.device_id == 0: 
                print(f"[PRE-TRAINING] Epoch {epoch} training loss:  {train_loss / len(train_loader)}")

        self.close_pretrain()
        
    def train(self):
        best_val_metrics = {}
        for epoch in range(self.num_epochs):
            train_loss = 0
            self.model.train()
            counts = 0
            for (x, y) in self.train_loader:
                self.optimizer.zero_grad() # reset gradients for new batch
                x = self.no_augment(x)
                x, y = x.to(self.device_id), y.to(self.device_id)
                if self.diffusion:
                    # denoise the labels for training of the original model
                    with torch.no_grad():
                        x_embed = self.model(x, encode_only=True)
                        y_clean = self.label_denoising_diffusion_model.module.reverse(x_embed, y).long()
                        
                    # print(f"noisy (original) label {y} | clean label {y_clean}")
                    y = y_clean
                    
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.detach().cpu().numpy()
                if self.device_id == 0 and counts % 200 == 0:
                    print(f"Main training loss idx {counts}: ", train_loss/(counts + 1))

                counts += 1
            
            if self.device_id == 0:
                print(f"[MAIN TRAINING] Training loss for epoch {epoch}: {train_loss / len(train_loader)}")
                counts = 0
                val_metrics = self.test()
            
            if (epoch == 0) or (self.device_id == 0 and val_metrics['test_f1_score'] > best_val_metrics['test_f1_score']):
                print(f"New best validation f1-score {val_metrics['test_f1_score']}, saving model.")
                best_val_metrics = val_metrics
                self.save_checkpoint(epoch, best_val_metrics, model_name='main_model')
            
def load_training_objects(batch_size, learning_rate, noise_type, bootstrap, beta, pretrain):
    
    os.makedirs('cifar_dataset', exist_ok=True)
    noise_path = '/home/faraz/person-detection-sota/review/cifar-10-100n/data/CIFAR-10_human.pt'
    train_dataset = CIFAR10(root='cifar_dataset', train=True, noise_type=noise_type, noise_path=noise_path, download=True)
    val_dataset = CIFAR10(root='cifar_dataset', train=False, noise_type='clean', download=True)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    model = ConvTwoLayerMLP(input_dim=3, num_classes=10, pretrain=pretrain)

    diffusion_params = {
        'img_emb_dim': 128,
        'num_classes': 10,
        'num_diffusion_steps': 1000
    }
    
    diffusion_model = LabelDenoisingDiffusionModel(
                x_dim=diffusion_params['num_classes'],
                x_emb_dim=diffusion_params['img_emb_dim'], 
                n_steps=diffusion_params['num_diffusion_steps']
    )
    
    # optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.0e-2)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    diffusion_optimizer = Adam(diffusion_model.parameters(), lr=learning_rate*0.1)
    
    loss = BootstrappedCrossEntropy(weight=None, ignore_index=-100, reduction='mean', bootstrap=bootstrap, beta=beta)
    optimizers = {
        'diffusion_optimizer': diffusion_optimizer,
        'main_optimizer': optimizer
        
    }
    return train_loader, val_loader, model, optimizers, loss, diffusion_model, diffusion_params

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

def set_seed(seed):
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed) # cuda
    torch.cuda.manual_seed_all(seed) # cuda multi-gpu
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # 
    torch.backends.cudnn.benchmark = False # no auto-optimization 
    torch.use_deterministic_algorithms(True)
    
# torchrun --nproc_per_node=4 noisy_train.py --batch_size 64 --lr 1e-3 --epochs 300 --noise_type worst  --checkpoint_dir checkpoints_diffusion --simclr --diffusion
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--batch_size', type=int, required=False, default=16, help='training batch size')
    arg_parser.add_argument('--lr', type=float, required=False, default=1.0e-3, help='training learning rate')
    arg_parser.add_argument('--epochs', type=int, required=False, default=10, help='how many epochs do we train this for')
    arg_parser.add_argument('--checkpoint_dir', type=str, required=False, default='./checkpoints/')
    arg_parser.add_argument('--noise_type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
    arg_parser.add_argument('--bootstrap', type=str, default='no_bootstrap')
    arg_parser.add_argument('--beta', type=float, default=0.8, help='weight for noisy labels')
    arg_parser.add_argument('--simclr', action='store_true', help='runs unsupervised pretraining using simclr before main training.')
    arg_parser.add_argument('--num_pretrain_epochs', type=int, default=50, help='number of epochs for pre-training of the encoder using simclr')
    arg_parser.add_argument('--diffusion', action='store_true', help='uses diffusion denoising for clean labels during training.')
    arg_parser.add_argument('--num_diffusion_epochs', type=int, default=50)
    
    arg_parser.add_argument('--test', action='store_true')
    args = arg_parser.parse_args()
    
    setup_ddp()
    device_ids = os.environ['LOCAL_RANK']

    num_epochs = args.epochs
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    noise_type = noise_type_map[args.noise_type]
    
    if args.bootstrap == 'soft':
        bootstrap = 'soft'
    elif args.bootstrap == 'hard':
        bootstrap = 'hard'
    else:
        bootstrap = 'no_bootstrap' 
        
    pretrain = True if args.simclr else False
    num_pretrain_epochs = args.num_pretrain_epochs
    train_loader, val_loader, model, optimizers, loss, diffusion_model, diffusion_params = load_training_objects(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        noise_type=noise_type,
        bootstrap=bootstrap,
        beta=args.beta,
        pretrain=pretrain
    )
    
    diffusion = True if args.diffusion else False
    num_diffusion_epochs = args.num_diffusion_epochs
    ddp_trainer = Trainer(
        diffusion=diffusion,
        diffusion_model=diffusion_model,
        diffusion_params=diffusion_params,
        num_diffusion_epochs=num_diffusion_epochs,
        pretrain=pretrain,
        num_pretrain_epochs=num_pretrain_epochs,
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        loss=loss,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir
    )
    
    torch.autograd.set_detect_anomaly(True)
    if args.test:
        ddp_trainer.load_checkpoint()
        ddp_trainer.test()
    else:
        if pretrain:
            # run pre-training (with simclr)
            ddp_trainer.run_pretrain()
            
        if diffusion:
            # use diffusion denoisiong (relies on embeddings from the simclr training)
            ddp_trainer.run_denoising_model_train()
            
        ddp_trainer.train()
        
    dist.destroy_process_group()







