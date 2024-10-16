from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_bev import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './lightning_logs/version_11797349/checkpoints/epoch=12-step=41495.ckpt'

# sd_locked = True  --> 12
# sd_locked = False --> 10
batch_size = 10

logger_freq = 300

# sd_locked = True  --> 1e-5
# sd_locked = False --> 2e-6
learning_rate = 2e-6
sd_locked = False
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
trainer = pl.Trainer(strategy="ddp", accelerator="gpu", gpus=3, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
