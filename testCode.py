
from myMath import myMath

print("start")
base = "sigmoid"
sig = getattr(myMath, base)
print(sig(34.))
