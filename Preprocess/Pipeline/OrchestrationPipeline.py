class OrchestrationPipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, data, verbose=True):
        for stage in self.stages:
            data = stage.process(data, verbose=verbose)
        return data
