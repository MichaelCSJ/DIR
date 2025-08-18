from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from options.option import Options
from data.dataset import *
from models.trainer import Trainer
from models.renderer import *
from models.reconstructor import *

def train():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # Option
    opt = Options().parse()
    # Tensorboard
    writer = SummaryWriter(opt.tb_dir)
    print('=================================================================================')
    print('tensorboard output: %s' % opt.tb_dir)
    print('=================================================================================')

    # Image formation & Reconstruction & Trainer
    optics_model = LightDeskRenderer(opt)
    recon_model = Reconstructor(opt)
    last_epoch = -1 if opt.load_step_start == 0 else opt.load_epoch
    trainer = Trainer(opt, optics_model, recon_model, writer, last_epoch)
    
    dataset_train = StereoDataset(opt)
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, num_workers=opt.num_threads, pin_memory=True)
    
    for epoch in range(1):
        print(f'Epoch: {epoch}')
        for data in tqdm(dataloader_train, desc=f"  Dataset", leave=False):
            trainer.run_model(data)
    writer.close()
    
if __name__=='__main__':
    train()