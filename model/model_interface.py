import inspect
import importlib

class MInterface(object):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config

    def load_model(self, **other_args):
        name = self.model_name
        camel_name = ''.join([i for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        model = self.instancialize(Model, self.config, **other_args)
        return model
    
    def instancialize(self, Model, config, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = other_args.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = other_args.get(arg, None)
        args1.update(**config)
        return Model(**args1)