# Coupled acceleration bounds

This additional verification case was run after the revision 2c.1 benchmark. It reproduces a false deterministic guarantee in the archived coordinator.

Let the plant satisfy x'' = w, with exact position measurements, zero initial state, a sampling period of 0.1 s and a true acceleration w = 1 m/s². Two allowed bounds on the same acceleration satisfy D + d = 2 and D,d ∈ [0,2]. The effective bound is min(D,d). It is zero at both vertices and one at the admissible midpoint D = d = 1. At t = 0.1 s the exact speed is therefore 0.1 m/s and the exact position is 0.005 m for this trajectory. The coordinator releases a zero-error bound and estimates the speed as 0.04649623235534 m/s. Its error of 0.05350376764466 m/s exceeds the requested limit of 0.001 m/s.

From the repository root, the following reads the exact request through the public interface. The original observations and gains are recorded in [coupled_bounds_observed.json](coupled_bounds_observed.json).

```python
import json
from pathlib import Path
from heros import api

case = Path("results/coupled_bounds_check/coupled_bounds_request.json")
request = json.loads(case.read_text())
answer = api.solve(**request)
t = request["t"][1]
estimate = float(answer["estimates"]["x_hat"][1][1])
print(answer["decision"]["disposition"], answer["decision"]["claim"]["value"])
print(t, estimate, abs(estimate - t))
```
