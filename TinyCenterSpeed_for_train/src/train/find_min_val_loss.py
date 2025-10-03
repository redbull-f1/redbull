import wandb
import pandas as pd

# 로그인 (이미 로그인되어 있다면 생략 가능)
wandb.login()

api = wandb.Api()
run = api.run("whdaudpark-dongguk-university/CenterSpeed/runs/f1812qv3")
# whdaudpark-dongguk-university/CenterSpeed/runs/f1812qv3

# val/loss 기록들을 DataFrame으로 가져오기
history = run.history(keys=["epoch", "val/loss"], pandas=True)

# 최소값과 이를 기록한 epoch 찾기
min_val_loss = history["val/loss"].min()
best_epoch = int(history.loc[history["val/loss"].idxmin(), "epoch"])

print(f"최저 val/loss: {min_val_loss:.4f} (epoch: {best_epoch})")
