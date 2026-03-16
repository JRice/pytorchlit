from pydantic import BaseModel, Field
from typing import List

# Given a list of integers nums and an integer target, find three integers in nums such that the sum is closest to target.
# Return the sum of the three integers. Only one solution need be found.

class ClosestMatch(BaseModel):
    # Ensure exactly 3 numbers are returned
    nums: List[int] = Field(..., min_length=3, max_length=3)
    sum_val: int
    distance: int

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
            
            # Have we found a closer sum?
            if abs(current_sum - target) < abs(best_sum - target):
                best_sum = current_sum
                best_nums = [nums[i], nums[left], nums[right]]

            if current_sum == target:
                break
            elif current_sum < target:
                left += 1
            else:
                right -= 1

        if best_sum == target:
            break

    return ClosestMatch(nums=best_nums, sum_val=best_sum, distance=abs(target - best_sum))

# Execution
nums = [-1, 2, 1, 4]
target = 6
result = three_sum_closest(nums, target)

print(f"Match: {result.nums}")
print(f"Total: {result.sum_val} (Distance: {result.distance})")
print(f"JSON Output: {result.model_dump_json()}")