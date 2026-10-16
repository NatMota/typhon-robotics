"""INIT12 command-only acquisition rule; no environment, task ID or model access."""
import math
from init11_contract import history

class CommandAcquisition:
    def __init__(self, enabled, initial_gripper_command):
        assert initial_gripper_command == -1., "Every request must document an open executed command"
        self.enabled = bool(enabled)
        self.last_gripper_command = float(initial_gripper_command)
        self.executed = 0
        self.modified = 0
        self.query_index = 0
        self.trigger = None
        self.pending = None
        self.pending_count = 0

    def prepare(self, raw):
        assert len(raw) == 30 and all(len(a) == 20 for a in raw)
        assert self.pending is None or self.pending_count == 30
        assert all(math.isfinite(float(v)) for a in raw for v in a)
        closing = next((i for i, a in enumerate(raw) if a[9] > .5), None)
        if self.enabled and self.trigger is None and self.last_gripper_command == -1. and closing is not None:
            self.trigger = {"query_zero_based": self.query_index,
                            "policy_action_zero_based": self.executed,
                            "first_closing_waypoint_zero_based": closing}
        active = self.trigger is not None and self.modified < 60
        issued = [[float(v) for v in a] for a in raw]
        if active:
            for a in issued: a[2] -= .005
        self.pending = {"query_zero_based": self.query_index, "modified": active,
                        "first_closing_waypoint_zero_based": closing,
                        "last_executed_gripper_command": self.last_gripper_command,
                        "policy_action_start_zero_based": self.executed}
        self.pending_count = 0
        self.query_index += 1
        return issued, dict(self.pending)

    def executed_action(self, issued, actual_gripper_command):
        assert self.pending is not None and self.pending_count < 30
        assert actual_gripper_command == (1. if issued[9] > .5 else -1.)
        self.last_gripper_command = float(actual_gripper_command)
        self.executed += 1
        self.pending_count += 1
        if self.pending["modified"]: self.modified += 1
        assert self.modified <= 60
        return history(issued)

    def summary(self):
        return {"enabled": self.enabled, "trigger": self.trigger,
                "policy_actions_executed": self.executed,
                "modified_actions_executed": self.modified,
                "modified_action_interval_zero_based_half_open":
                    None if self.trigger is None else
                    [self.trigger["policy_action_zero_based"],
                     self.trigger["policy_action_zero_based"] + self.modified],
                "last_executed_gripper_command": self.last_gripper_command}

def self_check():
    base = [[.1, -.2, .9, 1., 0., 0., 0., 1., 0., .5] + [0.] * 10 for _ in range(30)]
    close = [a.copy() for a in base]; close[17][9] = .500001
    off = CommandAcquisition(False, -1.)
    for raw in [base, close, close]:
        issued, event = off.prepare(raw)
        assert issued == raw and not event["modified"]
        for a in issued: assert off.executed_action(a, 1. if a[9] > .5 else -1.) == history(a)
    assert off.trigger is None and off.modified == 0
    on = CommandAcquisition(True, -1.)
    issued, event = on.prepare(base)
    assert issued == base and not event["modified"]
    for a in issued: on.executed_action(a, -1.)
    for q, raw in enumerate([close, close, base, close]):
        issued, event = on.prepare(raw)
        assert event["modified"] == (q < 2)
        for a, b in zip(issued, raw):
            assert all(a[j] == b[j] for j in range(20) if j != 2)
            assert abs((a[2] - b[2]) - (-.005 if q < 2 else 0.)) < 1e-15
            h = on.executed_action(a, 1. if a[9] > .5 else -1.)
            assert h == history(a) and h[9:] == [0.] * 11
    assert on.trigger == {"query_zero_based": 1, "policy_action_zero_based": 30, "first_closing_waypoint_zero_based": 17}
    assert on.modified == 60 and on.summary()["modified_action_interval_zero_based_half_open"] == [30, 90]
    late = CommandAcquisition(True, -1.)
    issued, event = late.prepare(close)
    for a in issued[:8]: late.executed_action(a, 1. if a[9] > .5 else -1.)
    assert late.modified == 8
    try: CommandAcquisition(True, 1.)
    except AssertionError: pass
    else: raise AssertionError("Closed initial command accepted")
    return {"passed": True, "checks": ["no-closing pass-through", "first close threshold >0.5",
        "pre-close waypoints included", "two chunks/60 actions", "no retrigger",
        "active Z only", "issued float32 first-nine/eleven-zero history",
        "disabled exact pass-through", "early horizon duration", "explicit open initialization"]}

if __name__ == "__main__":
    import json
    print(json.dumps(self_check()))
