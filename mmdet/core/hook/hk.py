from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(
            self,
            interval=-1,
            by_epoch=True,
            save_optimizer=True,
            out_dir=None,
            max_keep_ckpts=-1,
            **kwargs):
        for i in range(3):
            print("我被初始化了, __init__")

    def before_run(self, runner):
        for i in range(3):
            print("我被触发了, before_run")
