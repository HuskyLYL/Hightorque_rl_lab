class A:
    def __init__(self):
        print("A.__init__ 开始")
        self.load()
        print("A.__init__ 结束")
    
    def load(self):
        print("执行 A.load")
        

class a:  # 注意：小写a
    def __init__(self):
        print("a.__init__ 开始")
        self.load()
        print("a.__init__ 结束")
    
    def load(self):
        print("执行 a.load")

class B(A, a):  # B继承自A和a
    def __init__(self):
        print("B.__init__ 开始")
        super().__init__()  # 只会调用MRO中的下一个
        print("B.__init__ 结束")
    
    def load(self):
        print("执行 B.load")

class C(B):  # C继承自B
    def __init__(self):
        print("C.__init__ 开始")
        super().__init__()
        print("C.__init__ 结束")
    
    def load(self):
        print("执行 C.load")
        super().load()

# 测试
c = C()
print("\nMRO顺序：", C.__mro__)