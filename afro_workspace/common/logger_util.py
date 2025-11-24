import heapq

class LargestKRecorder:
    def __init__(self, K):
        self.scalars = []
        self.K = K

    def record(self, scalar):
        if len(self.scalars) < self.K:
            heapq.heappush(self.scalars, scalar)
        else:
            # Compare the new scalar with the smallest value in the heap
            if scalar > self.scalars[0]:
                heapq.heappushpop(self.scalars, scalar)

    def average_of_largest_K(self):
        if len(self.scalars) == 0:
            raise ValueError("No scalars have been recorded yet.")
        
        return sum(self.scalars) / len(self.scalars)

