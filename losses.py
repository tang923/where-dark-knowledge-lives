import torch
import torch.nn.functional as F


def kd_parts(student, teacher, target, temperature=4.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher = teacher.detach()
    s, t = student / temperature, teacher / temperature
    mask = F.one_hot(target, s.shape[1]).bool()
    s_other = s[~mask].reshape(len(target), -1)
    t_other = t[~mask].reshape(len(target), -1)
    s_binary = torch.stack((s[mask], s_other.logsumexp(1)), dim=1)
    t_binary = torch.stack((t[mask], t_other.logsumexp(1)), dim=1)

    def kl(a, b):
        return F.kl_div(F.log_softmax(a, 1), F.log_softmax(b, 1),
                        log_target=True, reduction="none").sum(1) * temperature**2

    return {"kd": kl(s, t), "tckd": kl(s_binary, t_binary),
            "nckd": kl(s_other, t_other),
            "other_probability": F.softmax(t_binary, 1)[:, 1]}


def objective(student, target, teacher=None, *, method="ce", temperature=4.0,
              ce_weight=1.0, distill_weight=1.0, alpha=1.0, beta=2.0):
    ce = F.cross_entropy(student, target)
    distill = student.new_zeros(())
    if method != "ce":
        if teacher is None:
            raise ValueError("Distillation requires a frozen teacher")
        if method == "kd":
            distill = F.kl_div(F.log_softmax(student / temperature, 1),
                              F.log_softmax(teacher.detach() / temperature, 1),
                              log_target=True, reduction="batchmean") * temperature**2
        else:
            parts = kd_parts(student, teacher, target, temperature)
            if method == "tckd":
                distill = alpha * parts["tckd"].mean()
            elif method == "nckd":
                distill = beta * parts["nckd"].mean()
            elif method == "dkd":
                distill = (alpha * parts["tckd"] + beta * parts["nckd"]).mean()
            else:
                raise ValueError(f"Unknown method: {method}")
    return ce_weight * ce + distill_weight * distill, ce.detach(), distill.detach()
