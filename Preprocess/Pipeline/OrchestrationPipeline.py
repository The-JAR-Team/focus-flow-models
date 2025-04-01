# OrchestrationPipeline.py
class OrchestrationPipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, data):
        for stage in self.stages:
            data = stage.process(data)
        return data
