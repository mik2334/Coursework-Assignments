class Node:

    def __init__(self, index=0, value=0, next=None):
        self.index = index
        self.value = value
        self.next = next    

class SparseVector:

    def __init__(self, length):
        self.length = length
        self.head = Node(-1, 0, None)
        self.size = 0

    # is_empty method added for explanation, not required due to sentinel node
    def is_empty(self):
        return self.size == 0
    
    def __len__(self):
        return self.length
    
    def get_element(self, index):
        assert 0 <= index < self.length
        curr = self.head.next
        while curr and curr.index <= index:
            if curr.index == index:
                return curr.value
            curr = curr.next
        return 0.0
        
    def add_node(self, index, value):
        assert isinstance(index, int), "index must be of integer value"
        assert isinstance(value, (int,float)), "element must be of real type"
        assert 0 <= index < self.length, "index must be within list length"
        
        if value == 0:
            return 
        
        prev = self.head
        curr = self.head.next

        while curr and curr.index < index:
            prev = curr
            curr = curr.next

        if curr and curr.index == index:
            curr.value = value
            return
        
        new_node = Node(index, value, curr)
        prev.next = new_node
        self.size += 1
    
    def __add__(self, other):
        assert self.length == other.length, "both vectors must be of the same length"

        result = SparseVector(self.length)
        p = self.head.next
        q = other.head.next

        while p and q:
            if p.index < q.index:
                result.add_node(p.index, p.value)
                p = p.next
            elif p.index > q.index:
                result.add_node(q.index, q.value)
                q = q.next
            else:
                sum = p.value + q.value
                if sum != 0:
                    result.add_node(p.index, sum)
                p = p.next
                q = q.next

        while p:
            result.add_node(p.index, p.value)
            p = p.next

        while q:
            result.add_node(q.index, q.value)
            q = q.next

        return result
    
    def __sub__(self, other):
        assert self.length == other.length, "both vectors must be of the same length"

        result = SparseVector(self.length)
        p = self.head.next
        q = other.head.next

        while p and q:
            if p.index < q.index:
                result.add_node(p.index, p.value)
                p = p.next

            elif p.index > q.index:
                result.add_node(q.index, -q.value)
                q = q.next

            else:
                sum = p.value - q.value
                if sum != 0:
                    result.add_node(p.index, sum)
                p = p.next
                q = q.next

        while p:
            result.add_node(p.index, p.value)
            p = p.next

        while q:
            result.add_node(q.index, -q.value)
            q = q.next

        return result
    
    def __mul__(self, other):
        assert self.length == other.length, "both vectors must be of the same length"

        result = 0
        p = self.head.next
        q = other.head.next

        while p and q:
            if p.index < q.index:
                p = p.next
            elif p.index > q.index:
                q = q.next
            else:
                result += p.value * q.value
                p = p.next 
                q = q.next

        return result


    def print(self):
        curr = self.head.next
        while curr:
            print(f"(index={curr.index}, value={curr.value})", end=" -> ")
            curr = curr.next
        print("None")