# 使用可能なCPUの数を確認
import os
import optuna  
from multiprocessing import Process
max_cpu = os.cpu_count()
max_cpu # 12

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2 

# DATABASE_URI = 'postgresql://{user}:{password}@{host}:{port}/{database_name}'
# DATABASE_URI = 'sqlite:///optuna_test1.db'
# study_name = 'example2_postgress'

# study = optuna.create_study(
#     study_name=study_name,
#     storage=DATABASE_URI,
#     load_if_exists=True
# )
# study.optimize(objective, n_trials=100)

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2 

def optimize(study_name, storage, n_trials):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

if __name__ == '__main__':
    # DATABASE_URI = 'postgresql://{user}:{password}@{host}:{port}/{database_name}'
    DATABASE_URI = 'sqlite:///optuna_test2.db'
    study_name = 'example3_distributed'

    n_trials = 300
    concurrency = 2  
    # max_cpuより使うCPUの数が多くないことを確認
    assert concurrency <= max_cpu
    n_trials_per_cpu = n_trials / concurrency


    # 並列化
    workers = [Process(target=optimize, args=(study_name, DATABASE_URI, n_trials_per_cpu)) for _ in range(concurrency)]
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()