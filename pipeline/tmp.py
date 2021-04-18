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
    def execute(self, prev_result=None):
        raise NotImplementedError


class Steps(Step):
    def __init__(self, steps: List[Step]):
        if not steps:
            steps = []

        self.__steps: List[Step] = steps

    def add_step(self, step: Step):
        self.__steps.append(step)

    def add_steps(self, steps: List[Step]):
        self.__steps.extend(steps)

    def clear_step(self):
        self.__steps.clear()

    def execute(self, prev_result=None):
        raise NotImplementedError


class Group(Steps):
    def execute(self, prev_result=None):
        return [step.execute(prev_result) for step in self.__steps]


class Serial(Steps):
    def execute(self, prev_result=None):
        for step in self.__steps:
            prev_result = step.execute(prev_result)
        return prev_result


class Adapter(Step):
    def parse(self, prev_result=None):
        raise NotImplementedError

    def execute(self, prev_result=None):
        return self.parse(prev_result=prev_result)


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

    def execute(self, prev_result=None):
        pass

