# Given a list of integers nums and an integer target, find three integers in nums such that the sum is closest to target.
# Return the sum of the three integers. Only one solution need be found.

def three_sum_closest(nums, target):
    nums.sort()
    # Even dwarves start small:
    best_triplet = nums[:3]
    best_sum = sum(best_triplet)
    
    for i in range(len(nums) - 2):
        # Skip duplicates to avoid redundant calculations
        if i > 0 and nums[i] == nums[i - 1]:
            continue
            
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == target:
                return { "nums": [nums[i], nums[left], nums[right]], "sum": current_sum }
            
            # Have we found a closer sum?
            if abs(current_sum - target) < abs(best_sum - target):
                best_sum = current_sum
                best_nums = [nums[i], nums[left], nums[right]]
                
            if current_sum < target:
                left += 1
            else:
                right -= 1
                
    return { "nums": best_nums, "sum": best_sum }

# Execution
nums = [-1, 2, 1, 4]
target = 6
result = three_sum_closest(nums, target)
print(f"The closest sum is {', '.join(map(str, result['nums']))} = {result['sum']}, which is {abs(target - result['sum'])} away from {target}.")