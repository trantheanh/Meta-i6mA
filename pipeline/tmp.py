from typing import List


class Executable:
    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def message(self, *args, **kwargs):
        return "{class_name}: \n{args} & \n{kwargs}".format(
            class_name=self.__class__.__name__,
            args=args,
            kwargs=kwargs
        )


class Step(Executable):
    def execute(self, _input=None):
        raise NotImplementedError


class Steps(Step):
    def __init__(self, *args: Step):
        self.__steps: List[Step] = [step for step in args]

    def add_step(self, step: Step):
        self.__steps.append(step)

    def add_steps(self, steps: List[Step]):
        self.__steps.extend(steps)

    def clear_step(self):
        self.__steps.clear()

    def execute(self, _input=None):
        raise NotImplementedError


class Group(Steps):
    def execute(self, _input=None):
        return [step.execute(_input) for step in self.__steps]


class Serial(Steps):
    def execute(self, _input=None):
        for step in self.__steps:
            _input = step.execute(_input)
        return _input


class Adapter(Step):
    def parse(self, _input=None):
        raise NotImplementedError

    def execute(self, _input=None):
        return self.parse(_input=_input)


class DeRequest(Step):
    def __init__(self, n_retry=1):
        self.n_retry = n_retry

    @property
    def url(self) -> str:
        raise NotImplementedError

    @property
    def body(self):
        raise NotImplementedError

    def on_success(self):
        raise NotImplementedError

    def on_fail(self):
        raise NotImplementedError

    def on_exception(self):
        raise NotImplementedError

    def execute(self, _input=None):
        pass



