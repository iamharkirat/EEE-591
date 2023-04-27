import numpy as np

def compute_pi(precision, max_points=10000):
    successful_attempts = 0
    avg_pi = 0
    
    for _ in range(100):
        inside_circle = 0
        total_points = 0
        
        for _ in range(max_points):
            x, y = np.random.uniform(0, 1), np.random.uniform(0, 1)
            r = np.sqrt(x**2 + y**2)
            
            if r <= 1:
                inside_circle += 1
                
            total_points += 1
            pi_estimate = 4 * (inside_circle / total_points)
            
            if abs(pi_estimate - np.pi) < precision:
                successful_attempts += 1
                avg_pi += pi_estimate
                break
                
    if successful_attempts > 0:
        avg_pi /= successful_attempts
        
    return successful_attempts, avg_pi

precisions = [10**(-i) for i in range(1, 8)]

for p in precisions:
    success, avg_pi = compute_pi(p)
    
    if success > 0:
        print(f"{p} success {success} times {avg_pi}")
    else:
        print(f"{p} no success")
