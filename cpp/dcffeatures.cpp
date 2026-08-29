// dcffeatures.cpp —— Y 量化 DCT 系数的盲隐写分析特征（ctypes 调用）
//
// 与 yccstego/steganalysis.py 的 analyze_y 逐项对齐，用于跨实现(BER/特征)一致性验证。
// 计算 (输入为量化 Y 网格，块主序 R×C×8×8 int16)：
//   out[0] n_ac           非零 AC 系数个数
//   out[1] parity_odd     AC 系数 LSB=1 的占比
//   out[2] unit_frac      |AC|==1 的占比（nsF5/F5 magnitude-1 指纹）
//   out[3] chi_stat       Westfeld 卡方统计量（相邻幅值对 (2k,2k+1)）
//   out[4] chi_p          卡方生存函数 p 值
//   out[5] df             卡方自由度
//   out[6] ac_plane_entropy  量化 AC 网格相邻差分熵（axis_bins=32，去直流）
//
// 编译 (MinGW)：
//   g++ -O2 -shared -o dcffeatures.dll dcffeatures.cpp -static
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

extern "C" {

// ---------- 不完全伽马（译自 steganalysis.py 的 _gser/_gcf） ----------
static double llgamma(double a) { return std::lgamma(a); }

static double gser(double a, double x, int itmax = 200, double eps = 3e-14) {
    if (a <= 0.0) return 1.0;
    if (x <= 0.0) return 0.0;
    double ap = a, s = 1.0 / a, dl = 1.0 / a;
    for (int i = 0; i < itmax; ++i) {
        ap += 1.0;
        dl *= x / ap;
        s += dl;
        if (std::fabs(dl) < std::fabs(s) * eps) break;
    }
    return s * std::exp(-x + a * std::log(x) - llgamma(a));
}

static double gcf(double a, double x, int itmax = 200, double eps = 3e-14,
                  double fpmin = 1e-300) {
    if (a <= 0.0) return 0.0;
    double b = x + 1.0 - a;
    double c = 1.0 / fpmin, d = 1.0 / b, h = 1.0 / b;
    for (int i = 1; i <= itmax; ++i) {
        double an = -double(i) * double(i - (long long)a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < fpmin) d = fpmin;
        c = b + an / c;
        if (std::fabs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        double dl = d * c;
        h *= dl;
        if (std::fabs(dl - 1.0) < eps) break;
    }
    return std::exp(-x + a * std::log(x) - llgamma(a)) * h;
}

static double gamma_q(double a, double x) {
    return (x < a + 1.0) ? 1.0 - gser(a, x) : gcf(a, x);
}

static double chi2_sf(double x, double df) {
    return gamma_q(df / 2.0, x / 2.0);
}

// ---------- 主分析 ----------
// Y: 量化 Y 网格，块主序，长度 R*C*64；R/C 为块的行/列数。
void dcffeatures_analyze_y(const int16_t* Y, int R, int C,
                           double* out /* 至少 7 */) {
    const int nblk = R * C;
    const int total = nblk * 64;

    // 1) 非零 AC 系数
    int n_ac = 0;
    long long odd_cnt = 0, unit_cnt = 0;
    long long mag_max = 0;
    std::vector<int> mags;
    mags.reserve(total);
    for (int k = 0; k < total; ++k) {
        if (k % 64 == 0) continue;         // DC
        int64_t v = Y[k];
        if (v == 0) continue;              // 零
        ++n_ac;
        if (v & 1) ++odd_cnt;
        long long av = v < 0 ? -(long long)v : (long long)v;
        if (av == 1) ++unit_cnt;
        if (av > mag_max) mag_max = av;
        mags.push_back((int)av);
    }
    double parity = n_ac ? double(odd_cnt) / n_ac : 0.0;
    double unit_frac = n_ac ? double(unit_cnt) / n_ac : 0.0;

    // 2) Westfeld 卡方：mag 的 paired (2k,2k+1)
    long long hi = mag_max + 1;
    if (hi < 64) hi = 64;
    std::vector<double> counts(hi + 1 + 1, 0.0);   // +1 保证偶数长度
    for (int m : mags) counts[m] += 1.0;
    long long len = counts.size() & 1 ? counts.size() + 1 : counts.size();
    counts.resize((size_t)len, 0.0);
    double chi_stat = 0.0, chi_p = 0.0;
    long long df = 0;
    long long ns = 0;
    long long half = len / 2;
    for (long long k = 0; k < half; ++k) {
        double even = counts[2 * k], odd = counts[2 * k + 1];
        double sum = even + odd;
        if (sum > 0.0) {
            double d = even - odd;
            chi_stat += d * d / sum;
            ++ns;
        }
    }
    if (ns == 0) {
        chi_p = 0.0; df = 0;
    } else if (ns <= 1) {
        chi_p = 1.0; df = 0;
    } else {
        df = ns - 1;
        chi_p = chi2_sf(chi_stat, (double)df);
    }

    // 3) 量化 AC 网格"展平再按 R*8×C*8 切行"矩阵的相邻差分熵（axis_bins=32, 去 DC）。
    //    必须复刻 steganalysis._plane_diff_entropy2 的 np.reshape(R*8,C*8):
    //    它把 (R,C,8,8) 展平(block主序, 块内行主序)直接切成每行 C*8 个元素, 并非图像空间拼图。
    //    val(k) = (k%64==0)? 0 : Y[k], k 为展平下标。
    const int W = C * 8, H = R * 8;
    auto val = [&](long long k) -> long long {
        return (k % 64 == 0) ? 0LL : (long long)Y[k];
    };
    std::vector<double> hist(32, 0.0);
    auto add_diff = [&](long long a, long long b) {
        long long d = a - b;
        if (d < 0) d = -d;
        if (d > 31) d = 31;
        hist[(size_t)d] += 1.0;
    };
    for (int gr = 0; gr < H; ++gr) {
        long long base = (long long)gr * W;
        for (int gc = 0; gc < W - 1; ++gc)
            add_diff(val(base + gc), val(base + gc + 1));
    }
    for (int gc = 0; gc < W; ++gc) {
        for (int gr = 0; gr < H - 1; ++gr)
            add_diff(val((long long)gr * W + gc), val((long long)(gr + 1) * W + gc));
    }
    double tot = 0.0;
    for (double x : hist) tot += x;
    double entropy = 0.0;
    if (tot > 0.0) {
        for (double c : hist) {
            if (c > 0.0) {
                double p = c / tot;
                entropy -= p * std::log2(p);
            }
        }
    }

    out[0] = (double)n_ac;
    out[1] = parity;
    out[2] = unit_frac;
    out[3] = chi_stat;
    out[4] = chi_p;
    out[5] = (double)df;
    out[6] = entropy;
}

} // extern "C"