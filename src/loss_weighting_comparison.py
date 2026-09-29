from src.newton_benchmark import compare_newton_performance


def compare_loss_weighting(previous_run, output_dir, steps=1000,
                           checkpoint_steps=(0, 100, 300, 500, 750, 1000),
                           search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                           data_dir='/content/data/', device='cuda'):
    return compare_newton_performance(previous_run=previous_run, output_dir=output_dir,
        steps=steps, checkpoint_steps=checkpoint_steps, search_seeds=search_seeds,
        final_seeds=final_seeds, compare_uniform=True, compare_fast=False,
        data_dir=data_dir, device=device)
