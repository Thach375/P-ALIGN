# Reward-Weighted Prefix-Alignment Distillation

*Cải tiến bước chọn dữ liệu của P-ALIGN bằng outcome-reward contrast (lấy cảm hứng từ OGLS-SD), giữ regime offline + SFT, kèm pha DPO tùy chọn để tương phản good/bad một cách ổn định.*

---

## 1. Tóm tắt một câu

Thay cơ chế lọc nhị phân của P-ALIGN bằng một cơ chế **chấm điểm theo nhóm**: cho student sinh nhiều bản trả lời từ cùng một prefix, chấm bằng verifier, chuẩn hóa thành advantage tương đối trong nhóm, rồi dùng advantage làm **trọng số** cho SFT thường. Nếu cần tương phản mạnh hơn, thêm một **pha DPO** trên các cặp (tốt, tệ) lấy từ chính các nhóm đó — thay cho ý tưởng unlikelihood (L3 cũ) vốn dễ gây bất ổn.

---

## 2. Vấn đề cần giải

P-ALIGN dựng dữ liệu bằng quy tắc lọc nhị phân: student sinh **một** bản tiếp nối từ prefix, giữ nếu đáp án khớp, bỏ nếu sai. Ba điểm yếu:

1. **Nhị phân và lãng phí.** Câu khó mà bản duy nhất sai thì vứt cả cặp (câu hỏi, prefix) — mất luôn prefix đã tốn công tìm. Đúng những câu khó là chỗ cần coverage nhất thì lại dễ bị loại.
2. **Không phân biệt chất lượng.** Hai bản cùng đúng được đối xử như nhau, kể cả khi một bản suy luận sạch còn một bản đúng nhờ may.
3. **Không tận dụng bản sai.** Tín hiệu "cái gì phân biệt bản đúng với bản sai" lại chính là thứ có thể khai thác.

---

## 3. Nền tảng: P-ALIGN

P-ALIGN distill long-chain reasoning từ teacher mạnh (DeepSeek-R1) vào student nhỏ, qua hai ý:

- **Adaptive prefix truncation.** Long CoT của teacher thường dài và nhiễu ở các bước về sau. P-ALIGN cắt trace thành câu, dùng **binary search** kết hợp **student tự đánh giá** (`ENOUGH` / `NOT_ENOUGH`) để tìm prefix **tối thiểu nhưng đủ** giải bài.
- **Prefix-based alignment.** Lấy prefix làm prior context, cho student tự sinh tiếp phần còn lại, dùng chuỗi `prefix + continuation` làm tín hiệu SFT.

Phương pháp này **giữ nguyên toàn bộ phần đó** và chỉ thay bước lọc tạo dữ liệu, thêm một pha huấn luyện tùy chọn.

---

## 4. Nguồn cảm hứng: OGLS-SD — và ranh giới mượn

OGLS-SD dùng **verifiable outcome reward để tương phản trajectory thành công và thất bại**, từ đó hiệu chỉnh tín hiệu distillation.

Ranh giới:

- **Mượn:** ý tưởng "dùng outcome reward để contrast bản đúng/bản sai trong một nhóm".
- **KHÔNG mượn:** cơ chế **logit steering** ở mức phân phối token. Logit steering cần logit của teacher và chung tokenizer/vocab, kèm vòng on-policy — những thứ làm nó nặng và không tương thích với cặp teacher/student khác vocab của P-ALIGN.

Ý reward-contrast được hiện thực ở **mức dữ liệu** (trọng số SFT) và, nếu muốn, ở **mức cặp** (DPO). Cả hai đều giữ tính nhẹ: teacher chỉ cần text, không cần logit, không on-policy.

---

## 5. Ý tưởng cốt lõi

> Cho student sinh **K** bản tiếp nối từ prefix thay vì 1. Chấm điểm từng bản bằng verifier so với đáp án chuẩn. Chuẩn hóa điểm thành **advantage tương đối trong nhóm** của cùng một câu. Dùng advantage làm **trọng số** cho SFT thường trên chuỗi `prefix + continuation`. Tùy chọn: thêm pha **DPO** trên các cặp (tốt, tệ) trong nhóm để tương phản mạnh hơn mà vẫn ổn định.

Hệ quả phụ: tỉ lệ đúng của nhóm K bản là **bằng chứng thực nghiệm** về chất lượng prefix, cho phép hiệu chỉnh ranh giới cắt prefix (chữa đúng limitation P-ALIGN tự nêu về self-judge của model nhỏ).

---

## 6. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $q^i$ | câu hỏi thứ $i$ |
| $a^i_*$ | đáp án chuẩn của câu $i$ |
| $R^i$ | long CoT đầy đủ của teacher |
| $\tilde{R}^i$ | prefix tối thiểu đủ (sau binary search) |
| $K$ | số bản tiếp nối sinh ra cho mỗi câu |
| $y^i_k$ | bản tiếp nối thứ $k$ |
| $\text{Ans}(\cdot)$ | hàm trích đáp án cuối |
| $r^i_k$ | reward của bản $k$ |
| $A^i_k$ | advantage của bản $k$ |
| $s^i_k = \tilde{R}^i \oplus y^i_k$ | chuỗi supervision (prefix nối continuation) |
| $\pi_\theta,\ \pi_{\text{ref}}$ | model student đang train; reference model (cho DPO) |

---

# PHA A — Dựng dữ liệu (offline)

## Bước 1 — Chuẩn bị teacher traces *(giữ nguyên)*

Dataset gốc gồm (câu hỏi, long CoT của teacher, đáp án chuẩn). Nếu dùng s1K-1.1 thì trace có sẵn, không cần gọi teacher lại; chỉ cần đảm bảo có đáp án chuẩn cho verifier. Teacher chỉ cung cấp **text**.

## Bước 2 — Adaptive prefix truncation *(giữ nguyên)*

Cắt trace thành câu, binary search trên số câu, mỗi bước cho student tự đánh giá prefix đã `ENOUGH` chưa. Hội tụ sau số lần gọi cỡ logarit theo số câu, cho ra prefix tối thiểu đủ $\tilde{R}^i$.

## Bước 3 — Sinh K continuation *(ĐỔI)*

Với mỗi $(q^i, \tilde{R}^i)$, student sinh $K$ bản tiếp nối **có điều kiện trên prefix**:

- Sinh với temperature $\approx 0.7$–$1.0$ và top-p $\approx 0.9$–$0.95$ để $K$ bản thực sự khác nhau (temperature quá thấp → các bản trùng nhau, không có gì để tương phản).
- Tín hiệu supervision về sau là chuỗi ghép $s^i_k = \tilde{R}^i \oplus y^i_k$.
- Kiểm tra đa dạng ngay lúc sinh: đếm số đáp án cuối khác nhau hoặc distinct-n; nếu nhóm sụp về giống hệt → nâng temperature hoặc rút ngắn prefix.

## Bước 4 — Outcome reward *(MỚI)*

Chấm từng bản bằng verifier so với **đáp án chuẩn** (không phải so với CoT teacher).

Bản nhị phân:

$$r^i_k = \mathbb{1}\!\left[\text{Ans}(y^i_k) = a^i_*\right]$$

Bản có phạt độ dài (ưu tiên bản đúng *và* ngắn, làm tường minh điều P-ALIGN làm ngầm qua cắt prefix):

$$r^i_k = \mathbb{1}\!\left[\text{Ans}(y^i_k) = a^i_*\right] - \lambda \cdot \mathrm{len}(y^i_k), \qquad \lambda \text{ nhỏ}$$

Với toán, hàm chấm là exact-match / `sympy` trên đáp án trong `\boxed{}`. Reward này là **phép đo chất lượng** để các bước sau biết giữ gì và học mạnh đến đâu — tính một lần, offline, không phải reward của RL.

## Bước 5 — Group-relative advantage *(MỚI)*

Trung bình và độ lệch chuẩn trong nhóm $K$ bản của **cùng một câu**:

$$\mu^i = \frac{1}{K}\sum_{k=1}^{K} r^i_k, \qquad \sigma^i = \sqrt{\frac{1}{K}\sum_{k=1}^{K}\left(r^i_k - \mu^i\right)^2}$$

Advantage của từng bản:

$$A^i_k = \frac{r^i_k - \mu^i}{\sigma^i + \epsilon}$$

Chuẩn hóa **theo từng câu** cho một hệ quả quan trọng: bản đúng ở **câu khó** (pass-rate thấp) tự động nhận advantage lớn, bản đúng ở **câu dễ** (pass-rate cao) nhận advantage nhỏ. Với reward nhị phân, advantage của một bản đúng là

$$A_{\text{đúng}} = \sqrt{\frac{1-p}{p}}, \qquad p = \text{pass-rate}$$

nên $p=0.1$ cho $A\approx 3.0$ còn $p=0.9$ cho $A\approx 0.33$ — thành công hiếm ở câu khó được nhấn mạnh tự động, đúng tín hiệu giá trị nhất để học.

## Bước 6 — Prefix-feedback loop *(MỚI, là điểm novelty)*

Tỉ lệ đúng của nhóm:

$$p^i = \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}\!\left[\text{Ans}(y^i_k) = a^i_*\right]$$

Quy tắc hiệu chỉnh prefix:

$$
\tilde{R}^i \leftarrow
\begin{cases}
\text{nới dài hơn} & \text{nếu } p^i < \tau_{\text{low}}\\[2pt]
\text{cắt ngắn hơn} & \text{nếu } p^i \approx 1 \text{ và } |\tilde{R}^i| \text{ đang nhỏ}
\end{cases}
$$

Đặt cap số vòng. Vòng này vừa cứu câu khó (vốn hay bị lọc nhị phân loại nhầm), vừa chữa đúng limitation P-ALIGN tự nêu: self-judge phụ thuộc năng lực model, model nhỏ dễ đánh giá sai sufficiency.

### Bốn trường hợp của một nhóm K bản

| Trường hợp | Mô tả | Xử lý |
|---|---|---|
| **Mixed** | có cả đúng lẫn sai | advantage cho contrast tự nhiên; cũng là nguồn cặp DPO — trường hợp lý tưởng |
| **All correct** | tất cả đúng, $\sigma=0$ | không contrast nhưng đều tốt → trọng số bằng nhau; nếu có phạt độ dài thì vẫn rank được theo độ ngắn; không tạo được cặp DPO |
| **All wrong** | tất cả sai | không có tín hiệu dương → bỏ khỏi SFT, gắn cờ cho prefix-feedback (Bước 6) |
| **K = 1** | suy biến | trở về đúng quy tắc lọc nhị phân gốc của P-ALIGN |

---

# PHA B — Huấn luyện

## Bước 7 — Reward-weighted SFT *(pha chính, thay quy tắc lọc nhị phân)*

Trọng số thô của mỗi bản, hai mức:

$$
w^i_k =
\begin{cases}
\mathbb{1}\!\left[r^i_k \ge \mu^i\right] & \text{(L1: lọc mềm — giữ bản trên trung bình nhóm)}\\[4pt]
\mathrm{clip}(A^i_k,\, 0,\, c) = \min\!\big(\max(A^i_k, 0),\, c\big) & \text{(L2: có thứ bậc theo advantage)}
\end{cases}
$$

Chuẩn hóa trọng số trong nhóm để scale loss ổn định:

$$\hat{w}^i_k = \frac{w^i_k}{\sum_{j=1}^{K} w^i_j}$$

Hàm loss (NLL có trọng số, chia độ dài để bản dài không lấn át):

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{i}\sum_{k=1}^{K} \hat{w}^i_k \cdot \frac{1}{|s^i_k|}\sum_{t=1}^{|s^i_k|} \log \pi_\theta\!\left(s^i_{k,t} \,\middle|\, s^i_{k,<t},\, q^i\right)$$

Lưu ý phân biệt khả năng của hai mức: **L1** cho mọi bản đúng trọng số bằng nhau (không ưu tiên câu khó); **L2** mới có hành vi "đúng hiếm → trọng số cao" nhờ advantage. Bắt đầu với L1, nâng lên L2. Optimizer giữ nguyên P-ALIGN: 3 epoch, lr $5\times10^{-5}$, LoRA. Checkpoint sau pha này gọi là $\pi_{\text{SFT}}$.

## Bước 8 — DPO refinement *(pha tương phản, tùy chọn — thay cho L3 cũ)*

Sau khi SFT cho một base tốt, có thể thêm một pha **DPO** để mài sắc khoảng cách giữa suy luận tốt và xấu, dùng các cặp lấy từ chính các nhóm K bản. DPO thay thế ý unlikelihood (L3 cũ) vì nó tương phản good/bad nhưng có sẵn **reference-model KL** làm trust region.

### Dựng cặp ưu tiên

Với mỗi nhóm **mixed**, tạo cặp $(y_w, y_l)$:

- $y_w$ (chosen) = bản reward cao nhất (đúng, ngắn nhất nếu có shaping).
- $y_l$ (rejected) = bản reward thấp (sai; hoặc đúng-nhưng-tệ hơn nhiều nếu muốn dạy conciseness).
- Chỉ chọn cặp có **khoảng cách reward rõ** để tránh cặp nhiễu. Thường 1 cặp/nhóm (best vs worst) là đủ.
- Bỏ qua nhóm all-correct và all-wrong (không có contrast).

### Hàm loss DPO

Với reference $\pi_{\text{ref}} = \pi_{\text{SFT}}$ (đóng băng):

$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\left[\log \sigma\!\left(\beta\Big[\big(\log \pi_\theta(y_w\!\mid\! x) - \log \pi_{\text{ref}}(y_w\!\mid\! x)\big) - \big(\log \pi_\theta(y_l\!\mid\! x) - \log \pi_{\text{ref}}(y_l\!\mid\! x)\big)\Big]\right)\right]$$

### Vì sao DPO hợp với setup này hơn L3

- **Prefix tự triệt tiêu.** Vì cả chosen và rejected đều mang cùng prefix $\tilde{R}$, số hạng prefix khử nhau trong hiệu log-prob:

$$\log\pi_\theta(\tilde{R}\oplus y_w\mid x) - \log\pi_\theta(\tilde{R}\oplus y_l\mid x) = \log\pi_\theta(y_w\mid x,\tilde{R}) - \log\pi_\theta(y_l\mid x,\tilde{R})$$

  Nên DPO chỉ học từ **phần continuation khác nhau**, không bao giờ dìm token prefix tốt — đúng cái bẫy đã khiến L3 nguy hiểm thì DPO né tự động (không cần mask tay).
- **Có trust region.** Reference $\pi_{\text{SFT}}$ giữ $\pi_\theta$ không trôi xa — đây là cái L3 (weighted-MLE âm) hoàn toàn thiếu.

### Lưu ý khi dùng DPO

- **Chất lượng cặp quyết định tất cả.** Dùng cùng verifier reward để chọn cặp; ưu tiên khoảng cách reward lớn.
- **Likelihood displacement.** DPO đôi khi kéo tụt cả log-prob của chosen. Ghìm lại bằng $\beta$ vừa phải (thường $\approx 0.1$), reference tốt, không train quá lâu, hoặc thêm một số hạng NLL phụ trên chosen (DPO+SFT) cho ổn định.
- **Bộ nhớ reference.** Có thể **precompute** log-prob của $\pi_{\text{ref}}$ trên các cặp một lần rồi cache, để khi train DPO không phải giữ reference model trong bộ nhớ.

---

## 9. Vì sao thiết kế này hợp lý

- **Cứu câu khó.** Lọc nhị phân loại cặp ngay khi bản duy nhất sai; sinh $K$ bản + chuẩn hóa theo câu giúp vớt được sample dùng được, giữ coverage trên phần khó.
- **Nhấn vào thành công hiếm.** Group-relative advantage tự cho bản đúng ở câu khó trọng số cao (L2) — tín hiệu giá trị nhất.
- **Bắt false positive.** Khi reward có tín hiệu độ dài/format, bản đúng-nhờ-may nhận điểm thấp hơn bản đúng-thật.
- **Dùng được bản sai.** Bản sai đặt mốc cho advantage, và là vế "rejected" cho DPO.
- **Tương phản ổn định.** DPO thay L3 cho contrast mạnh mà có trust region; prefix tự khử khỏi tín hiệu.

---

## 10. Định vị trong literature *(để defend)*

Cơ chế group-relative advantage là phần lõi của GRPO, nhưng phương pháp này **không phải GRPO**: không policy gradient, không importance ratio, không on-policy ở pha SFT. Framing đúng:

> *P-ALIGN làm backbone (prefix truncation + prefix-alignment + SFT target); thay filter nhị phân bằng data selection kiểu rejection-sampling FT với group-relative advantage của GRPO; lấy cảm hứng từ ý tương phản success/failure của OGLS-SD; thêm pha DPO cho contrast ổn định; đóng góp mới là vòng prefix-feedback.*

Citation cần có: P-ALIGN; OGLS-SD; GRPO (DeepSeekMath, Shao et al. 2024) cho advantage; họ RAFT/ReST/STaR cho multi-sample + weighting; DPO (Rafailov et al. 2023) cho pha tương phản. (Kiểm lại năm/tên chính xác khi đưa vào bibliography.)

### Provenance từng bước

| Bước | Dòng dõi |
|---|---|
| 3 sinh K bản | RFT family (STaR/RAFT/ReST/GRPO), self-consistency |
| 4 outcome reward | correctness từ P-ALIGN (Eq. 9) / RLVR; framing contrast từ OGLS-SD; shaping là chung |
| 5 group advantage | GRPO |
| 6 prefix-feedback | **mới** (đóng góp gốc) |
| 7 weighted SFT | target từ P-ALIGN; weighting từ RAFT/RWR |
| 8 DPO | DPO (Rafailov et al.); SFT→DPO là recipe chuẩn (Zephyr/Tulu) |

---

## 11. Thiết kế thí nghiệm

Bậc thang ablation, mỗi nấc cô lập một đóng góp:

1. **P-ALIGN gốc** (lọc nhị phân, $K=1$) — baseline.
2. **L1** ($K>1$) vs baseline — multi-sample + giữ-trên-trung-bình có giúp không.
3. **L2** vs L1 — weighting theo advantage có thêm lợi không.
4. **Quét $K$** ($1,4,8$) — phần lợi từ multi-sample.
5. **Bật/tắt prefix-feedback (Bước 6)** — đo riêng; lý tưởng cho thấy coverage trên câu khó (AIME) tăng.
6. **+ DPO (Bước 8)** vs chỉ SFT — pha tương phản có đẩy thêm không.
7. *(tùy chọn)* reward có/không phạt độ dài — conciseness có rút ngắn trace mà giữ accuracy không.

---

## 12. Rủi ro và lưu ý

- **Reward hacking độ dài.** $\lambda$ quá mạnh khiến model cụt suy luận. Giữ nhỏ, theo dõi độ dài cùng accuracy.
- **Nhóm suy biến.** Xử lý rõ all-correct/all-wrong; đừng chia cho $\sigma=0$.
- **Trọng số bản đúng hiếm quá cao (L2).** Bản đúng-nhờ-may ở câu rất khó có thể bị kéo mạnh — trần clip $c$ và shaping giữ lại.
- **Chi phí sinh mẫu.** Sinh $K$ bản nhân chi phí generation lên $K$ lần, nhưng là rollout student (không cần logit teacher), offline, song song được; $K=4$–$8$ hợp lý.
- **DPO.** Phụ thuộc chất lượng cặp; coi chừng likelihood displacement; cache reference log-prob để tiết kiệm bộ nhớ; không train quá lâu (overoptimization).
- **Phụ thuộc verifier.** Với toán exact-match/`sympy` đủ tốt; miền khác cần verifier riêng.

---

## 13. Pseudocode tổng

```text
# ---- PHA A: dựng dữ liệu ----
for (q, R, a*) in D:
    R̃ = adaptive_prefix(q, R)                       # Bước 2 (giữ nguyên)
    for attempt in range(T_max):
        Y = [sample(π_θ, InstructAlign(q, R̃), temp=τ_s, top_p=0.95) for _ in range(K)]  # Bước 3
        r = [verify(y, a*) - λ * len_penalty(y) for y in Y]                              # Bước 4
        p = mean(is_correct(y, a*) for y in Y)
        if p < τ_low:
            R̃ = extend_prefix(q, R); continue        # Bước 6 (prefix-feedback)
        break
    if all_wrong(r):
        continue
    A = (r - mean(r)) / (std(r) + ε)                 # Bước 5
    for k in range(K):
        w = weight(A[k], r[k], mean(r))              # L1 / L2
        sft_data.add((q, R̃ ⊕ Y[k]), weight=w)        # Bước 7
    if mixed(r):                                     # dựng cặp DPO
        dpo_pairs.add(prompt=q, prefix=R̃,
                      chosen=argmax_reward(Y, r),
                      rejected=low_reward(Y, r))

# ---- PHA B: huấn luyện ----
normalize_weights_per_group(sft_data)
π_SFT = train_SFT(π_θ, sft_data, weighted_NLL, LoRA, epochs=3, lr=5e-5)   # Bước 7
# tùy chọn:
π_final = train_DPO(π_SFT, dpo_pairs, ref=π_SFT, beta=0.1)                 # Bước 8
```

**Defaults khởi động:** $K = 4$–$8$, temperature $\tau_s = 0.7$–$1.0$, top-p $0.95$, $\epsilon = 10^{-6}$, $\tau_{\text{low}} \approx 0.2$, $\lambda$ nhỏ, weight bắt đầu từ L1; DPO $\beta \approx 0.1$.
