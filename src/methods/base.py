from pathlib import Path


class BaseMethod:
    def __init__(self, params, output_dir=None):
        self.params = params
        self.output_dir = Path(output_dir) if output_dir else None

    def setup(self):
        """Heavy init (models, tokenizers). Called once. Resume from checkpoints if present."""
        pass

    def run(self, data, seed):
        """Train (or load) and return a results dict."""
        raise NotImplementedError

    def evaluate(self, data, seed, dataset_name):
        """Readouts of the trained seed. One checkpoint per seed for every method except those
        whose output is several policies (em_minmax_dpo), which override this."""
        from evaluation import evaluate_checkpoint
        return evaluate_checkpoint(self.output_dir / f"seed{seed}" / "model", data, self.cfg, dataset_name,
                                   params=self.params)

    def teardown(self):
        pass
