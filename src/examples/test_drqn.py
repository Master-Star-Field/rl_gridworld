import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from src.rl_grid_world.envs.grid_world import GridWorldEnv
from src.rl_grid_world.agents.drqn_agent import DRQN_Lightning

mask = np.zeros((6, 6))
mask[2, 1:5] = 1 

env_config = {
        "h": 6, 
        "w": 6,
        "n_colors": 2,
        "pos_goal": (5, 5),
        "pos_agent": (0, 0),
        "obstacle_mask": mask,
        "render_mode": "rgb_array"
    }

model = DRQN_Lightning(
        env_params=env_config,
        seq_len=8,
        batch_size=32,
        lr=2e-3,
        epsilon_decay=3000
)

trainer = pl.Trainer(
        max_steps=10000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=50
)

print("Начинаем обучение DRQN (с памятью LSTM)...")
trainer.fit(model)
    
print("Тестирование...")
model.eval()
test_env = GridWorldEnv(**env_config, render_mode="human")
    
state, _ = test_env.reset()
hidden = None 
done = False
    
while not done:
    test_env.render()
        
    state_t = torch.FloatTensor(state).view(1, 1, -1).to(model.device)
        
    with torch.no_grad():
        q_values, hidden = model.q_net(state_t, hidden)
        action = q_values.argmax(dim=2).item()
            
    state, reward, term, trunc, _ = test_env.step(action)
    done = term or trunc
    plt.pause(0.2)