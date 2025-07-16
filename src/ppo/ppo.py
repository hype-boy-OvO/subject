from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from src.env.env import CustomEnv
from src.ppo.custom.evalcallback import CustomEvalCallback

env = make_vec_env(lambda: CustomEnv(), n_envs=8)
eval_env = make_vec_env(lambda: CustomEnv(render_mode="rgb_array"), n_envs=8, seed=42) 

eval_env = VecVideoRecorder(
    eval_env,
    video_folder="src/ppo/logs/videos/",                    
    record_video_trigger=lambda step: step%200 == 0,  
    video_length=200,                           
    name_prefix="eval_2"                         
)
        
eval_callback = CustomEvalCallback(
    stdlimit=50,
    eval_env=eval_env,
    best_model_save_path="src/ppo/logs/best_model/",
    log_path="src/ppo/logs/results/",
    eval_freq=128,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)



model = RecurrentPPO(
    policy="MultiInputLstmPolicy",
    env=env,
    verbose=1,
    device="cuda",
    batch_size=32,
    n_steps=4
)




model = RecurrentPPO.load("src/ppo/logs/best_model/best_model.zip", env=env, device="cuda")
model.learn(total_timesteps=1e15, callback=[eval_callback])