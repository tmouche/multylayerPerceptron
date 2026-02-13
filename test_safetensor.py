
# import inspect

# def logger(msg):
#     frame = inspect.currentframe().f_back
#     func_name = frame.f_code.co_name

#     object_name = ""
#     try:
#         object_name = frame.f_locals['self'].__class__.__qualname__
#     except:
#         object_name = "<module>"
#     print(f"[TEST][{object_name}.{func_name}]-> {msg}")


# def not_in():
#     logger("hello")

# class Test:

#     def test_fonction(self):
#         logger("test")

#     @property
#     def __where__(self) -> str:
#         frame = inspect.currentframe().f_back
#         return f"{self.__class__.__qualname__}.{frame.f_code.co_name}"
    

# tt = Test()
# tt.test_fonction()
# not_in()

mother = [0,0,0]
if not mother:
    print(mother)