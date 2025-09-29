

class test:
    @staticmethod
    def testFunc(msg):
        print(f"oui c est moi {msg}")

good = getattr(test, "testFunc")
good("hector")
try:
    wrong = getattr(test, "testFunction")
    wrong("Achille")
except AttributeError:
    print("Catchowww ")
