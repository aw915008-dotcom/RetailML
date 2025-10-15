import subprocess
import time

# Define your apps and their file names
apps = [
    "SentimentAnalysisAPI.py",
    "ProductPredictionAPI.py",
    "DemandForecastingAPI.py",
    "CustomerSegAPI.py",
    "CustomerClassfiAPI.py",
    "BranchPredictionAPI.py",
    "GateWay.py"
]

processes = []

try:
    # Start each Flask app in a separate background process
    for app_file in apps:
        print(f"🚀 Starting {app_file} ...")
        p = subprocess.Popen(["python", app_file])
        processes.append(p)
        time.sleep(1)  # optional small delay between starts
    print("All Flask apps are now running.")

    # Keep the launcher alive (Ctrl+C to stop all)
    while True:
        time.sleep(60)

except KeyboardInterrupt:
    print("\n🛑 Stopping all Flask apps...")
    for p in processes:
        p.terminate()
    print("All stopped successfully.")
