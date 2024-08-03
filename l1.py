 for i in range(k):
            nums[nums.index(min(nums))]=-1*nums[nums.index(min(nums))]
        return sum(nums)
for i in range(k):
        nums[nums.indices(min(nums))]=-1*nums[nums.index(min(nums))]
    return sum(nums)
--------------------------------------------------------------------
x=""
for i in bin(n)[2:]:
    if i=0:
        x+="1"
    else:
        x+="0"
    return(x,2)
---------------------------------------------------------------------
if(n==1 or n==2):
    return 1
    a=1
    b=1
    sum=0
    mod=1000000007
for i in range(3,n+1):
   sum=(a+b)%mod
   a=b
   b=sum
return sum
-------------------------------------------------------------------------------
n=len(arr)-2
count={}
result=[]
for num in arr:
    if num in count:
        count+=1
    else:
        count=1
for num in arr:
    if count[num]==2:
        result.append(num)
return result
-----------------------------------------------------------------------------
count=1
c=1
k=n+m
for i in range(k,k+m,-1):
    count*1
for j in range(1,m+1):
    c*=j
return (c//count)%1000000007
-----------------------------------------------------------------------------
 mod=(10**9)+7
        dp=[[0]*(m+1)for _ in range(n+1)]
        for i in range(n+1):
            for j in range(m+1):
                if i==0 or j==0:
                    dp[i][j]=1
                else:
                    dp[i][j]=(dp[i-1][j]+dp[i][j-1])%mod
        return dp[n][m]
-----------------------------------------------------------------------------
 l=0
        r=0
        len(arr)-1
        while(l<=r):
            mid=(l+r)//2
            if(arr[mid]<=ele):
                l=mid+1
            else:
               r=mid-1
        return r
        b_sorted=sorted(b)
        result=[]
        for x in queries:
            idx = binary_search(b_sorted,a[x])
            result.append(b_sorted[:idx+1])
    return result
------------------------------------------------------------------------------
class Solution:
    def deleteK(self, head, k):
        #code here
        first=node(0)
        first.next=head
        n,w=0,first
        while w:
            n=(n+1)%k
            if n==0 and w.next:
                w.next=w.next.next
            else:
                w=w.next
        return first.next
    -------------------------------------------------------------------------
  current.node=head
        
        prev.node=null
        Node n=null
        while(current!=null):
            n=current.next
            current.next=prev
            prev=current
            current=n
        return prev
        Node n1=reverse(num1)
        Node n2=reverse(num2)
        Node ans=new Node(-1)
        Node re=ans
        carry=0
        while(n1!=null||n1!=null):
            sum=carry
            if(n1!=null):
                sum+=n1.data
                n1=n1.next
            if(n2!=null):
                sum+=n2.data
                n2=n2.next
            Node temp=new Node(sum%10)
            ans.next=temp
            ans=temp
            carry=sum/10
        if(carry!=0):
            Node temp=new Node(carry)
            ans.next=temp
            ans=temp
        Node a=reverse(re.next)
        while(a!=null&&a.data==0):
            a=a.next
        if(a==null):
            return new Node(0)
    return a
                
 ----------------------------------------------------------------------------
#vowels in word
 def arrangeCV(self, head):
    if not head:
        return None
    vowel_head = vowel_tail = Node(None)
    consonant_head = consonant_tail = Node(None)
    curr = head
    while curr:
        if curr.data in ['a', 'e', 'i', 'o', 'u']:
            vowel_tail.next = curr
            vowel_tail = vowel_tail.next
        else:
            consonant_tail.next = curr
            consonant_tail = consonant_tail.next
        curr = curr.next
    
    vowel_tail.next = consonant_head.next
    consonant_tail.next = None
    return vowel_head.next
-----------------------------------------------------------------------------
 #fraction given same or different
        fractions = str.split(',')
        fraction1 = fractions[0].split('/')
        fraction2 = fractions[1].split('/')
        num1,num2 = int(fraction1[0]),int(fraction2[0])
        den1,den2 = int(fraction1[1]),int(fraction2[1])
        if num1*den2 == num2*den1:
            return "equal"
        return fractions[0] if num1*den2 > num2*den1 else fractions[1].strip()
 ---------------------------------------------------------------------------------           
#printing braces
   class Solution:
	def bracketNumbers(self, str):
		# code here
		op_num = 0
		stack= []
		answer = []
		for char in str:
		    if char == '(':
		        op_num+=1
		        stack.append(op_num)
		        answer.append(op_num)
		    elif char == ')':
		        if stack:
		            answer.append(stack.pop())
		return answer
--------------------------------------------------------------------------------
#indexes of subarray sum
class Solution:
    def subArraySum(self,arr, n, s): 
       #Write your code here
       allsum,start=0,0
       for i in range(n):
           allsum+=arr[i]
           while(allsum>s and start<i):
               allsum-=arr[start]
               start=start+1
           if(allsum==s):
                return [start+1,i+1]
       return [-1]
-------------------------------------------------------------------------------
#summed matrix
class Solution:
    def sumMatrix(self, n, q):
        
        # code here 
        if(q>1 and q<2*n+1):
            if(q<=n+1):
                return q-1
            else:
                return n+1-(q-(n+1))-1
        else:
            return 0
-------------------------------------------------------------------------------
#missing array
class Solution:
    
    # Note that the size of the array is n-1
    def missingNumber(self, n, arr):
        
        # code here
        return n*(n+1)//2-sum(arr)
---------------------------------------------------------------------------------
#second largest
class Solution:
	def print2largest(self,arr, n):
	    
		# code here
		if n < 2:
            return -1
        
        # Initialize first and second largest
        first = second = -1
        
        for i in range(n):
            if arr[i] > first:
                second = first
                first = arr[i]
            elif arr[i] > second and arr[i] != first:
                second = arr[i]
        
        return second
--------------------------------------------------------------------------------
#rotate matrix
class Solution:
    def rotateMatrix(self, k, mat):
        # code here
        x=k%len(mat[0])
        if x==0:
            return mat
        for i in range(len(mat)):
            mat[i]=mat[i][x:]+mat[i][:x]
        return mat
-------------------------------------------------------------------------------
#counting binary matrix(count0 in given matrix)
class Solution:
	def findCoverage(self, matrix):
		# Code here
		row=len(matrix)
		col=len(matrix[0])
		totalcount=0
		for i in range(row):
		    for j in range(col):
		        if matrix[i][j]==0:
		            count=0
		            if j>0 and matrix[i][j-1]==1:
		                count+=1
		            if j<(col-1) and matrix[i][j+1]==1:
		                count+=1
		            if i>0 and matrix[i-1][j]==1:
		                count+=1
		            if i<(row-1) and matrix[i+1][j]==1:
		                count+=1
		            totalcount+=count
		return totalcount
-------------------------------------------------------------------------------
#max of subarray
def maxProduct(self, arr, n):
    if n == 0:
        return 0
        
    # Initialize maximum, minimum product for the current position and result
    max_product = arr[0]
    min_product = arr[0]
    result = arr[0]

    for i in range(1, n):
        if arr[i] == 0:
            max_product = 1
            min_product = 1
            result = max(result, 0)
            continue

        temp_max = arr[i] * max_product
        temp_min = arr[i] * min_product

        max_product = max(temp_max, temp_min, arr[i])
        min_product = min(temp_max, temp_min, arr[i])

        result = max(result, max_product)

    return result
 ------------------------------------------------------------------------------
 #diagonal matrix
 def isToeplitz(mat):
    #code here
    n=len(mat)
    m=len(mat[0])
    if(m==1):
        return True
        
    for i in range(1,n):
        for j in range(1,m):
            if(mat[i-1][j-1]!=mat[i][j]):
                return False
    return True
-------------------------------------------------------------------------------
#polindrome matrix
def isRp(self,i,arr):
        s,e=0,len(arr[0])-1
        while s<e:
            if arr[i][s]!=arr[i][e]:
                return False
            s+=1
            e-=1
        return True
    
    def isCp(self,i,arr):
        s,e=0,len(arr)-1
        while s<e:
            if arr[s][i]!=arr[e][i]:
                return False
            s+=1
            e-=1
        return True
-------------------------------------------------------------------------------
#comparing two linkedlists
current1 = head1
    current2 = head2
    
    # Traverse both lists
    while current1 is not None and current2 is not None:
        # Compare data at each node
        if current1.data != current2.data:
            return False
        
        # Move to the next node in both lists
        current1 = current1.next
        current2 = current2.next
    
    # Check if both lists ended at the same time
    if current1 is None and current2 is None:
        return True
    else:
        return False

# Helper function to create a linked list from a list of values
def create_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next
    return head
-------------------------------------------------------------------------------
#  building binary tree using linked list
 class ListNode:

    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.next = None


# Tree Node structure
class Tree:

    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None



#Function to make binary tree from linked list.
def convert(head):
    
  
    # code here
    a = []
    
    curr = head
    n=0
    while curr!=None:
        a.append(curr.data)
        curr = curr.next
        n += 1
        
    def buildTree(i):
        if i>=n:
            return
        newNode = Tree(a[i])
        left = buildTree(2*i+1)
        right = buildTree(2*i+2)
        newNode.left = left
        newNode.right = right
        return newNode
        
    root = buildTree(0)
    return root
-------------------------------------------------------------------------------
#linked as polindrome
class Node:
    def __init__(self,data):
        self.data=data
        self.next=None

def compute(head): 
    #return True/False
    temp=''
    temp+=head.data
    head=head.next
    while head:
        temp+=head.data
        head=head.next
    if temp==temp[::-1]:
        return True
    return False
-------------------------------------------------------------------------------
#remove duplicate from linked list
 class Node :
        def __init__(self, val):
            self.data = val
            self.next = None
class Solution:
    def removeAllDuplicates(self, head):
        #code here
        arr = []
        curr = head
        while curr!=None:
            arr.append(curr.data)
            curr = curr.next
            
        dp = [0]*(max(arr)+1)
        for i in range(len(arr)):
            dp[arr[i]]+=1
            
        dummy = Node(0)
        dummy.next = head
        prev = dummy
        curr = head
        while curr:
            if dp[curr.data] > 1:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
    
        return dummy.next
--------------------------------------------------------------------------------
#duplicate subtrees
class Solution:
    def printAllDups(self, root):
        # code here
        ret={}
        seen=set()
        def dfs(cur=root):
            if not cur:
                return [-1]
            left=dfs(cur.left)
            right=dfs(cur.right)
            tmp=[cur.data]+left+right
            if tuple(tmp) in seen:
                ret[tuple(tmp)]=cur
            else:
                seen.add(tuple(tmp))
            return tmp
        dfs()
        return [ret[x] for x in sorted(ret)]
--------------------------------------------------------------------------------
# verticalWidthofBinaryTree
def verticalWidth(root):
    # code here
    if not root:
        return 0

    min_pos = max_pos = 0  
    def traverse(node, pos):
      nonlocal min_pos, max_pos 
      if not node:
        return
      min_pos = min(min_pos, pos)  
      max_pos = max(max_pos, pos)  
      traverse(node.left, pos - 1)  
      traverse(node.right, pos + 1)  

    traverse(root, 0)  
    return max_pos - min_pos + 1

--------------------------------------------------------------------------------
#populate Inorder Successor
class Node:

    def __init__(self, val):
        self.right = None
        self.data = val
        self.left = None
        self.next = None


class Solution:

    def populateNext(self, root):
        if not root:
            return
        
        # Initialize an empty stack and a pointer for the current node
        stack = []
        current = root
        prev = None  # Previous node in the in-order traversal

        while stack or current:
            # Reach the leftmost node of the current node
            while current:
                stack.append(current)
                current = current.left
            
            # Current is now the leftmost node
            current = stack.pop()
            
            # If there was a previous node, set its next pointer to current
            if prev:
                prev.next = current
            
            # Update prev to current
            prev = current
            
            # Visit the right subtree
            current = current.right

-------------------------------------------------------------------------------
#ancestors in binary tree
class Node:
    def __init__(self,val):
        self.data = val
        self.left = None
        self.right = None
class Solution:
    def Ancestors(self, root, target):
        '''
        :param root: root of the given tree.
        :return: None, print the space separated post ancestors of given target., don't print new line
        '''
        #code here
        def find_ancestors(node, target, path):
            if not node:
                return False
            # Add current node to path
            path.append(node.data)
            # If current node is the target, return True
            if node.data == target:
                return True
            # Recur for left and right subtree
            if (find_ancestors(node.left, target, path) or 
                find_ancestors(node.right, target, path)):
                return True
            # If not found in either left or right subtree, backtrack
            path.pop()
            return False
        
        # List to store the path (ancestors)
        path = []
        # Call the helper function with the initial path
        find_ancestors(root, target, path)
        # Since the target itself will be included, we remove it
        if path and path[-1] == target:
            path.pop()
        # The ancestors should be in reverse order
        path.reverse()
        return path
--------------------------------------------------------------------------------
#searching sorted array
class Solution:
    def search(self,arr,key):
        # Complete this function
        for i in range(0,len(arr)):
            if arr[i] == key:
                return i
        return -1   


-------------------------------------------------------------------------------
#closet three sums
class Solution:
    def threeSumClosest (self, arr, target):
    # Your Code Here
        arr.sort()
        n = len(arr)
        min_diff = float('inf')
        ans = -float('inf')
        
        for i in range(n - 2):
            a,b = i + 1, n-1
            while a < b:
                y = arr[i] + arr[a] + arr[b]
                x = abs(y - target)
                
                if x == min_diff:
                    ans = max(ans, y)
                elif min_diff > x:
                    min_diff = x
                    ans = y
                
                if y > target:
                    b -= 1
                else:
                    a += 1
        return ans
--------------------------------------------------------------------------------
#largest square formed in a matrix
from typing import List


class Solution:
    def maxSquare(self, n : int, m : int, mat : List[List[int]]) -> int:
        # code here
        mxs=0
        for i in range(n):
            for j in range(m):
                if mat[i][j]==1:mxs=max(mxs,1)
                if i and j:
                    if mat[i-1][j-1] and mat[i-1][j] and mat[i][j-1] and mat[i][j]:
                        mat[i][j]=min(mat[i-1][j-1],mat[i-1][j],mat[i][j-1])+1
                        mxs=max(mxs,mat[i][j])
        return mxs
--------------------------------------------------------------------------------
#maximum connected group
  def dfs(x, y, index):
            # Stack-based DFS to avoid recursion limit issues
            stack = [(x, y)]
            size = 0
            while stack:
                i, j = stack.pop()
                if visited[i][j]:
                    continue
                visited[i][j] = True
                size += 1
                component[i][j] = index
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and not visited[ni][nj] and grid[ni][nj] == 1:
                        stack.append((ni, nj))
            return size

        n = len(grid)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = [[False] * n for _ in range(n)]
        component = [[-1] * n for _ in range(n)]
        component_size = []
        
        # Step 1: Find all connected components and their sizes
        index = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1 and not visited[i][j]:
                    size = dfs(i, j, index)
                    component_size.append(size)
                    index += 1
        
        max_size = max(component_size, default=0)
        
        # Step 2: Try changing each 0 to 1 and compute the resulting connected component size
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    seen_components = set()
                    size_after_change = 1
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] == 1:
                            comp_idx = component[ni][nj]
                            if comp_idx not in seen_components:
                                size_after_change += component_size[comp_idx]
                                seen_components.add(comp_idx)
                    max_size = max(max_size, size_after_change)
        
        return max_size
---------------------------------------------------------------------------------
#sum of root to leaf nodes
class Node:
    def __init__(self,val):
        self.data = val
        self.left = None
        self.right = None



class Solution:

    def hasPathSum(self, root, target):
        '''
        :param root: root of given tree.
        :param sm: root to leaf sum
        :return: true or false
        '''
        # code here
        if not root:
            return False
        queue = []
        if root.left:
            queue.append([root.left, root.data + root.left.data])
        if root.right:
            queue.append([root.right, root.data + root.right.data])
        
        while queue:
            node = queue[0]
            queue = queue[1:]
            if node[1] == target and not (node[0].left or node[0].right):
                return True
            if node[0].left:
                queue.append([node[0].left, node[1] + node[0].left.data])
            if node[0].right:
                queue.append([node[0].right, node[1] + node[0].right.data])
        return False
--------------------------------------------------------------------------------
#shortest path of weighted undirected graph
from typing import List
class Solution:
    def shortestPath(self,n:int, m:int, edges:List[List[int]] )->List[int]:
        # code here
        from collections import defaultdict, deque
        from heapq import heappush, heappop
        
        g = defaultdict(list)
        for frm, to, w in edges:
            g[frm].append((to, w))
            g[to].append((frm, w))
        
        q = [(0, 1)]
        costs = {1: 0}
        parents = {1: None}

        
        def build_path(cost, last):
            nonlocal parents
            q = deque()
            while last:
                q.appendleft(last)
                last = parents[last]
            q.appendleft(cost)
            return q
            
        while q:
            cost0, frm = heappop(q)
            if frm == n:
                return build_path(cost0, frm)
            for to, w in g[frm]:
                cost = cost0+w
                if to not in costs or cost < costs[to]:
                    costs[to] = cost
                    parents[to] = frm
                    heappush(q, (cost, to))
        return [-1]
        
-------------------------------------------------------------------------------
#maximum connected group
from typing import List


class Solution:
    def MaxConnection(self, grid : List[List[int]]) -> int:
        # code here
        def dfs(x, y, index):
            # Stack-based DFS to avoid recursion limit issues
            stack = [(x, y)]
            size = 0
            while stack:
                i, j = stack.pop()
                if visited[i][j]:
                    continue
                visited[i][j] = True
                size += 1
                component[i][j] = index
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and not visited[ni][nj] and grid[ni][nj] == 1:
                        stack.append((ni, nj))
            return size

        n = len(grid)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = [[False] * n for _ in range(n)]
        component = [[-1] * n for _ in range(n)]
        component_size = []
        
        # Step 1: Find all connected components and their sizes
        index = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1 and not visited[i][j]:
                    size = dfs(i, j, index)
                    component_size.append(size)
                    index += 1
        
        max_size = max(component_size, default=0)
        
        # Step 2: Try changing each 0 to 1 and compute the resulting connected component size
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    seen_components = set()
                    size_after_change = 1
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] == 1:
                            comp_idx = component[ni][nj]
                            if comp_idx not in seen_components:
                                size_after_change += component_size[comp_idx]
                                seen_components.add(comp_idx)
                    max_size = max(max_size, size_after_change)
        
        return max_size
--------------------------------------------------------------------------------
#segregaye 0's and 1's
class Solution:
    def segregate0and1(self, arr):
        # code here
        i,j = 0,len(arr)-1
        
        while i < j:
            
            if arr[i] == 0: i += 1
            
            elif arr[j] == 1: j -= 1
            
            else: arr[i],arr[j] = arr[j],arr[i] 
            
        return arr
---------------------------------------------------------------------------------
#samllest number
class Solution:
    def smallestNumber(self, s, d):
        # code here
        def sum(n):
            res=0
            while n>0:
                res+=(n%10)
                n=n//10
            return res
        
        
        mini=10**(d-1)
        maxi=10**(d)
        
        for i in range(mini,maxi):
            if s==sum(i):
                return i
        return -1
--------------------------------------------------------------------------------
 #remaining string
class Solution:
	def printString(self, s, ch, count):
		# code here
		for i, c in enumerate(s):
            if c == ch: count -= 1
            if count == 0: return s[i+1:]
        return ""
--------------------------------------------------------------------------------
#construct a binary tree from parent array
class Node:
    # A utility function to create a new node
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class Solution:
    #Function to construct binary tree from parent array.
    def createTree(self,parent):
        # your code here
        n = len(parent)
        # create an arry to store created nodes
        created = [None]*n
        
        root = None
        
        for i in range(n):
            if created[i] is None:
                created[i] = Node(i)
            if parent[i] == -1:
                root = created[i]
            else:
                if created[parent[i]] is None:
                    created[parent[i]] = Node(parent[i])
                    
                p = created[parent[i]]
                if p.left is None:
                    p.left = created[i]
                else:
                    p.right = created[i]
        return root
--------------------------------------------------------------------------------
#longest alternating subsequence
class Solution:
    # Function to find the maximum length of alternating subsequence
    def alternatingMaxLength(self, arr):
       #code here
        incre=1
        decre=1
        for i in range(1,len(arr)):
            if arr[i]>arr[i-1]:
                incre=decre+1
            elif arr[i]<arr[i-1]:
                decre=incre+1
        return max(incre,decre)
-------------------------------------------------------------------------------
#count smaller element
 def update(bit, index, value):
            while index < len(bit):
                bit[index] += value
                index += index & -index
        
        def query(bit, index):
            sum = 0
            while index > 0:
                sum += bit[index]
                index -= index & -index
            return sum
        
        # Create a sorted copy of the array
        sorted_arr = sorted(arr)
        # Create a rank map to compress the values
        rank = {v: i+1 for i, v in enumerate(sorted_arr)}
        
        bit = [0] * (len(arr) + 1)
        result = []
        
        for num in reversed(arr):
            r = rank[num]
            result.append(query(bit, r - 1))
            update(bit, r, 1)
        
        return result[::-1]
----------------------------------------------------------------------------------
#remove half nodes
class Solution:
    def RemoveHalfNodes(self, node):
        #code here
        if not node or (not node.left and not node.right):
            return node
        node.left = self.RemoveHalfNodes(node.left)
        node.right = self.RemoveHalfNodes(node.right)
         # if left child doesn't exist grab right child and pass to parent and vice-versa
        if not node.left:  
            return node.right
        if not node.right:
            return node.left
        
        return node
--------------------------------------------------------------------------------
#maximum product subset of array
class Solution:
    def findMaxProduct(self, arr):
        # Write your code here
        ans, maxneg, mod = 1, float('-inf'), 10**9+7
        for i in arr:
            if i != 0:
                ans *= i
            if i < 0 and maxneg < i:
                maxneg = i
        if ans < 0:
            ans //= maxneg
        ans = ans%mod
        return ans
-------------------------------------------------------------------------------
#largest Bst
class Solution:
    def largestBst(self, root):
        # Your code here
        def helper(node):
            # Base case: If the node is None, return size=0, min=inf, max=-inf, and is_bst=True
            if not node:
                return (0, float('inf'), float('-inf'), True)
            
            # Recursively get the properties of left and right subtrees
            l_size, l_min, l_max, l_is_bst = helper(node.left)
            r_size, r_min, r_max, r_is_bst = helper(node.right)
            
            # Check if the current node is the root of a BST
            if l_is_bst and r_is_bst and l_max < node.data < r_min:
                size = 1 + l_size + r_size
                min_val = min(l_min, node.data)
                max_val = max(r_max, node.data)
                return (size, min_val, max_val, True)
            else:
                return (max(l_size, r_size), 0, 0, False)
        
        size, _, _, _ = helper(root)
        return size
-------------------------------------------------------------------------------
# merge bst's
 def dfs(node, arr):
            if not node:
                return
            dfs(node.left, arr)
            arr.append(node.data)
            dfs(node.right, arr)
            
        bst1, bst2 = [], []
        dfs(root1, bst1)
        dfs(root2, bst2)
        
        return sorted(bst1 + bst2)
--------------------------------------------------------------------------------
#check for BST
class Solution:
    
    #Function to check whether a Binary Tree is BST or not.
    def isBST(self, root):
        #code here
        val=-float('inf')
        def dfs(cur=root):
            nonlocal val
            if not cur:
                return True
            left=dfs(cur.left)
            if not left:
                return False
            if cur.data<=val:
                return False
            val=cur.data
            right=dfs(cur.right)
            if not right:
                return False
            return True
        return dfs()
--------------------------------------------------------------------------------
#array to bst
class Solution:
    def sortedArrayToBST(self, nums):
        # code here
        def fun(low,high):
            nonlocal root
            if low>high:
                return None
            mid=(low+high)//2
            ptr=Node(nums[mid])
            if root==None:
                root=ptr
            ptr.left=fun(low,mid-1)
            ptr.right=fun(mid+1,high)
            return ptr
        root=None
        fun(0,len(nums)-1)
        return root
--------------------------------------------------------------------------------
#K-pangrams
 class Solution:
    def kPangram(self,string, k):
    # code here
        string=string.split(" ")
        res=0
        for i in string:
            res+=len(i)
        if res<26:
            return False
        d={}
        for i in string:
            for j in i:
                d[j]=d.get(j,0)+1
        if len(d)==26:
            return True
        elif 26-len(d)<=k:
            return True
        return False
--------------------------------------------------------------------------------
#find a palindrome
 class Solution:
    def countMin(self, str):
        # code here
        n=len(str)
        dp= [[None]*(n+1) for _ in range(n+1)]
        for i in range(n+1):
            for j in range(n+1):
                if i==0 or j==0:
                    dp[i][j]=0
                else:
                    if str[i-1]==str[n-j]:
                        dp[i][j]=1+dp[i-1][j-1]
                    else:
                        dp[i][j]=max(dp[i-1][j],dp[i][j-1])
        return n-dp[n][n]
-------------------------------------------------------------------------------
#Rows with max 1's
class Solution:

	def rowWithMax1s(self,arr):
		# code here
		for j in range(len(arr[0])):
		    for i in range(len(arr)):
		        if arr[i][j]==1:
		            return i
	    return -1
--------------------------------------------------------------------------------
#rat in maze
from typing import List

class Solution:
    def findPath(self, m: List[List[int]]) -> List[str]:
        # code here
        n = len(m)
        if m[0][0] == 0 or m[n-1][n-1] == 0:
            return []

        def dfs(x, y, path, paths):
            if x == n - 1 and y == n - 1:
                paths.append(path)
                return

            # Mark the cell as visited
            m[x][y] = 0

            # Possible directions
            directions = [('D', 1, 0), ('L', 0, -1), ('R', 0, 1), ('U', -1, 0)]

            for dir in directions:
                move, dx, dy = dir
                nx, ny = x + dx, y + dy

                if 0 <= nx < n and 0 <= ny < n and m[nx][ny] == 1:
                    dfs(nx, ny, path + move, paths)

            # Unmark the cell (backtracking)
            m[x][y] = 1

        paths = []
        dfs(0, 0, '', paths)
        return sorted(paths)
--------------------------------------------------------------------------------
# longest common prefix to string
class Solution:
    def longestCommonPrefix(self, arr):
        # code here
        arr.sort()
        ans = ''
        flg = True
        for i in range(len(arr[0])):
            c = arr[0][i]
            for j in range(1,len(arr)):
                if arr[j][i]==c:
                    flg = True
                else:
                    flg = False
            if flg:
                ans +=c
            else:
                break
        if not ans:
            return '-1'
        return ans
--------------------------------------------------------------------------------
 #spirally traversing a matrix
class Solution:
    # Function to return a list of integers denoting spiral traversal of matrix.
    def spirallyTraverse(self, matrix):
        # code here
        l,r,t,b = 0,len(matrix[0])-1,0,len(matrix)-1
        result = []
        while l<=r and t<=b:
            for i in range(l,r+1):
                result.append(matrix[t][i])
            t+=1
            for i in range(t,b+1):
                result.append(matrix[i][r])
            r-=1
            if t<=b:
                for i in range(r,l-1,-1):
                    result.append(matrix[b][i])
                b-=1
            if l<=r:
                for i in range(b,t-1,-1):
                    result.append(matrix[i][l])
                l+=1
        return result
--------------------------------------------------------------------------------
#edit distance
class Solution:
	def editDistance(self, str1, str3):
		# Code here
		n = len(str1)
        m = len(str3)
        dp = {}
        def solve(i,j):
            if i<0:
                return j+1
            if j<0:
                return i+1
            
            if (i,j) in dp:
                return dp[(i,j)]
            
            if str1[i]==str3[j]:
                return 0+solve(i-1,j-1)
            
            insert = 1+solve(i,j-1)
            delete = 1+ solve(i-1,j)
            replace = 1+ solve(i-1,j-1)
            dp[(i,j)] = min(insert,delete,replace)
            return dp[(i,j)]
        return solve(n-1,m-1)
-------------------------------------------------------------------------------
#the celebrity problem
class Solution:
    def celebrity(self, mat):
        # code here
        n=len(mat)
        a=0
        b=n-1

        while a<b:
            if mat[a][b]==1:
                a+=1
            else:
                b-=1
        


        for i in range(n):
            if i!=a and (mat[a][i] ==1 or mat[i][a]==0):
                return -1
            
        return a
--------------------------------------------------------------------------------







    

        



        




 







   





