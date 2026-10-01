"""Turn pair probabilities into matches (validated in E-D1).

1. Exclusive assignment: each record keeps only its highest-probability S1.
2. Expected-F0.5 decoding: per S1, keep the top-k records that maximise
       E[F0.5] ~ 1.25 * sum_topk(p) / (k + 0.25 * (sum_all(p) + C))
   and predict nothing when P(no true match) = prod(1 - p) is higher (a singleton S1 scores 1 only
   when nothing is predicted). C = expected true matches the candidates missed.
"""
import polars as pl

EF_C, EF_FLOOR = 0.15, 0.05  # E-D1: M1 0.9420 -> 0.9429, M2 0.9471 -> 0.9482 (others within 0.0001)


def exclusive(scored: pl.DataFrame) -> pl.DataFrame:
    return scored.sort(["p", "s1"], descending=[True, False], maintain_order=True).group_by("rec", maintain_order=True).first()


def ef_decode(ex: pl.DataFrame, c: float = EF_C, floor: float = EF_FLOOR) -> pl.DataFrame:
    cols = ex.columns
    d = ex.filter(pl.col("p") >= floor).sort(["s1", "p", "rec"], descending=[False, True, False]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1").alias("_k"),
        pl.col("p").cum_sum().over("s1").alias("_P"),
        pl.col("p").sum().over("s1").alias("_E"),
        (1 - pl.col("p").clip(0, 1 - 1e-9)).log().sum().over("s1").exp().alias("_EF0"),
    ).with_columns((1.25 * pl.col("_P") / (pl.col("_k") + 0.25 * (pl.col("_E") + c))).alias("_EF"))
    d = d.with_columns(pl.col("_EF").max().over("s1").alias("_EFmax"),
                       pl.col("_k").filter(pl.col("_EF") == pl.col("_EF").max()).first().over("s1").alias("_kstar"))
    return d.filter((pl.col("_k") <= pl.col("_kstar")) & (pl.col("_EFmax") > pl.col("_EF0"))).select(cols)


if __name__ == "__main__":
    s = pl.DataFrame({"s1": ["a", "a", "a", "b", "c", "c"], "rec": ["r1", "r2", "r3", "r4", "r5", "r1"],
                      "p": [0.95, 0.90, 0.20, 0.30, 0.99, 0.10]})
    ex = exclusive(s)
    assert ex.filter(pl.col("rec") == "r1")["s1"].item() == "a"          # r1 goes to its best S1 only
    kept = set(ef_decode(ex).select("s1", "rec").rows())
    assert kept == {("a", "r1"), ("a", "r2"), ("c", "r5")}, kept       # weak a/r3 and lone b/r4 dropped
    print("decode self-check ok")
