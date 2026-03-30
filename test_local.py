import requests
import time

BASE_URL = "http://localhost:7860"

def run_dummy_task(task_level):
    print(f"\n{'='*40}\n🚀 STARTING DUMMY TEST: {task_level.upper()}\n{'='*40}")
    
    # 1. Reset the environment for this specific task
    requests.post(f"{BASE_URL}/reset", json={"task_level": task_level})
    requests.get(f"{BASE_URL}/state").json()

    # 2. Hardcoded perfect actions for each specific task
    if task_level == "easy":
        # Easy: Route is blocked, must go North -> South -> East
        actions = [
            {"action_type": "load_truck", "truck_id": "T101", "warehouse": "North", "amount": 20},
            {"action_type": "route_truck", "truck_id": "T101", "route_id": "North_to_South"},
            {"action_type": "wait", "hours": 5},
            {"action_type": "load_truck", "truck_id": "T101", "warehouse": "South", "amount": 20},
            {"action_type": "route_truck", "truck_id": "T101", "route_id": "South_to_East"},
            {"action_type": "wait", "hours": 8}
        ]
        
    elif task_level == "medium":
        # Medium: Move 50 units direct to East under a strict $300 budget
        actions = [
            {"action_type": "load_truck", "truck_id": "T101", "warehouse": "North", "amount": 50},
            {"action_type": "route_truck", "truck_id": "T101", "route_id": "North_to_East"},
            {"action_type": "wait", "hours": 10}
        ]
        
    # --- Hard branch of test_local.py ---
    elif task_level == "hard":
        actions = [
            # 1. T101 picks up VIP cargo at North
            {"action_type": "load_truck", "truck_id": "T101", "warehouse": "North", "amount": 40},
            {"action_type": "route_truck", "truck_id": "T101", "route_id": "North_to_East"},
            # 2. T102 is at South, must go to North to get Standard cargo
            {"action_type": "route_truck", "truck_id": "T102", "route_id": "South_to_North"},
            {"action_type": "wait", "hours": 5}, # T102 arrives North
            # 3. T102 loads and heads back to South
            {"action_type": "load_truck", "truck_id": "T102", "warehouse": "North", "amount": 10},
            {"action_type": "route_truck", "truck_id": "T102", "route_id": "North_to_South"},
            {"action_type": "wait", "hours": 10} # Both trucks arrive
        ]

    # 3. Execute the actions
    for step, action in enumerate(actions):
        print(f"\n--- Step {step + 1} ---")
        print(f"🤖 Sending Action: {action['action_type']} -> {action}")
        
        response = requests.post(f"{BASE_URL}/step", json=action).json()
        obs_message = response.get('observation', {}).get('message', 'No message')
        print(f"🌍 Server Reply: {obs_message}")
        
        if response.get('done'):
            print(f"\n✅ Server signaled {task_level.upper()} task is DONE!")
            break
            
        time.sleep(1)

    # 4. Check the final Grader score
    score_response = requests.get(f"{BASE_URL}/grader").json()
    score = score_response.get('score', 0.0)
    print(f"🏆 FINAL {task_level.upper()} SCORE: {score} / 1.0")
    return score

if __name__ == "__main__":
    # Loop through all three tasks!
    for t in ["easy", "medium", "hard"]:
        run_dummy_task(t)
        time.sleep(2) # Brief pause between tasks