primes=[2]

for x in range(3, 10000 + 1):
    is_prime=True
    if x<=1:
        continue
    for prime in primes:
        if x%prime==0:
            is_prime=False
            break
    if is_prime:
        primes.append(x)

print(primes)