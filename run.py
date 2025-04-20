from Preprocess.Pipeline.InspectData import inspect
from Preprocess.Pipeline.Pipelines.ConfigurablePipeline import run_threaded_pipeline
from Preprocess.Pipeline.Pipelines.TowProcessesAttempt import run_tow_processes_pipeline

# --- Configuration --- abs paths
BASE_RESULTS_DIR = r"C:\Users\bhbha\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\dataset\Cache\PipelineResult\10fps_quality95\01" # <<<--- ADJUST THIS PATH ---<<<
INSPECT_BATCH_SIZE = 32 # Batch size for full iteration
# -------------------

if __name__ == "__main__":
    config_file_path = "./Preprocess/Pipeline/Pipelines/configs/ENGAGENET_10fps_quality95_randdist.json"

    max_workers = 4  # Adjust as needed
    run_threaded_pipeline(config_file_path, max_workers)

    # run_tow_processes_pipeline(config_file_path)

    inspect(BASE_RESULTS_DIR, INSPECT_BATCH_SIZE)
