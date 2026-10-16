"""Pure-data INIT11 contract: no model, simulator, CUDA or optional dependencies."""
import math, struct
CASES = ("init-demo_0003-loading-912172", "init-demo_0005-loading-912171")
Q1_SHA = "26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a"
PROBE_SHA = "9ddf8971546fd494bfc56b53586e091e039a2c809e012931d21e5be519f552e5"
def history(last_target):
    assert len(last_target) == 20 and all(math.isfinite(float(x)) for x in last_target)
    return [struct.unpack("<f", struct.pack("<f", float(x)))[0] for x in last_target[:9]] + [0.] * 11
def query_budget():
    return [(q, min(30, 1436 - q * 30)) for q in range(4, 48)]
def self_check():
    a = [float(i) / 7 for i in range(20)]
    p = history(a)
    assert len(p) == 20 and p[9:] == [0.] * 11
    assert p[:9] == list(struct.unpack("<9f", struct.pack("<9f", *a[:9])))
    d = a.copy(); d[2] -= .005
    assert history(d)[2] != p[2]
    assert all(history(d)[i] == p[i] for i in range(20) if i != 2)
    budget = query_budget()
    assert len(budget) == 44 and sum(n for _, n in budget) == 1316
    assert budget[0] == (4, 30) and budget[-1] == (47, 26)
    assert 463 + 1316 + 24 == 1803 and 456 + 1316 + 24 == 1796
    try: history([float("nan")] * 20)
    except AssertionError: pass
    else: raise AssertionError("nonfinite target accepted")
    print("INIT11 contract self-check passed: masking, float32, modified Z, exact horizon.")
if __name__ == "__main__": self_check()
