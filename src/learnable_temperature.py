from src.newton_benchmark import compare_newton_performance


def compare_temperature(previous_run, output_dir, temperature_lr=.003, steps=1000,
                        checkpoint_steps=(0, 100, 300, 500, 750, 1000),
                        search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                        fast_solver=True, cg_check_interval=8,
                        data_dir='/content/data/', device='cuda'):
    if temperature_lr <= 0:
        raise ValueError('Temperature learning rate must be positive')
    return compare_newton_performance(previous_run=previous_run, output_dir=output_dir,
        temperature_lr=temperature_lr, steps=steps, checkpoint_steps=checkpoint_steps,
        search_seeds=search_seeds, final_seeds=final_seeds, compare_fast=fast_solver,
        cg_check_interval=cg_check_interval, data_dir=data_dir, device=device)
