---
type: note
kind: reference
title: DIP Project — equations for haze, water, glass, plastic and stacks (spatial and frequency domain)
---

# Equations for each medium

How each medium changes a picture, derived from the physics, written in the spatial domain and in the frequency domain, checked against the literature and against our own measurements. Code that implements them: `code/degrade/`. Checks: `code/experiments/run_frequency_analysis.py` (table `theory_checks.csv`), `validate_models.py`, `tune_models.py`, `check_scene_matched.py`. Findings are numbered F.. in [[Subjects/DIP/Projects/DIP-Project/log|log]]; the story is in [[Subjects/DIP/Projects/DIP-Project/docs/results-analysis|results-analysis]].

> [!important]
> **Notation.** $J(x)$ is the clear scene (what we want), $I(x)$ the picture we observe, $x$ a pixel position, $d(x)$ the distance from the camera, $f$ a spatial frequency in cycles per pixel, $\hat I(f)$ the Fourier transform of $I$. $t$ is the **transmission** (the fraction of scene light that gets through, 1 = nothing lost) and $A$ the light the medium adds. All pictures are in $[0,1]$.

## 0. One idea behind every medium

Every medium does two things to the light from the scene: it **keeps a fraction** and it **adds some light of its own**. As an equation, per colour channel:

$$ I = t\,J + A\,(1-t). $$

What differs between media is *where the numbers come from*: $t$ may depend on distance (haze, water), be uniform (plastic), or vary over the glass (film, droplets); $A$ may be white, blue-green or the colour of the film; and some media also **blur** ($J \to h * J$). The frequency domain then tells us what each does to the *scales* of a picture.

```mermaid
flowchart LR
    J["Clear scene J"] --> M1["Medium 1<br>keep t1, add A1"]
    M1 --> M2["Medium 2<br>keep t2, add A2"]
    M2 --> I["Observed picture I"]
    style I fill:#e6f4ea,stroke:#34a853,color:#000
```

*Read it as:* stacking media is just doing the same operation twice. The scene-side medium acts first and the camera-side medium acts last (section 5).

**Two facts from Fourier analysis are used everywhere.**
1. **Scaling:** multiplying a picture by a constant $t$ multiplies every frequency by $t$: $\widehat{tJ}(f)=t\hat J(f)$.
2. **A constant added to a picture only changes the zero frequency** (the average): $\widehat{c}(f) = c\,N\,\delta(f)$, with $N$ the number of pixels. A *slowly varying* addition changes only the lowest frequencies.

Convolution in space is multiplication in frequency, and multiplication in space is convolution in frequency: $\widehat{h*J}=\hat h\hat J$ and $\widehat{wJ}=\hat w\circledast\hat J$.

**The measured quantity.** The spectral ratio $R(f)=\big|\hat I(f)\big|\,/\,\big|\hat J(f)\big|$, averaged over rings of equal $f$, says how much of each spatial scale survives. A flat ratio at $t$ means "everything scaled by $t$"; a falling ratio means "fine detail lost".

---

## 1. Haze and fog

### Spatial domain

**Direct light fades with distance.** A beam of scene light of strength $E$ loses a fixed fraction per unit distance to scattering, $\mathrm{d}E/\mathrm{d}z=-\beta E$ (Beer–Lambert), so after distance $d$:

$$ E(d)=E_0\,e^{-\beta d}\quad\Rightarrow\quad t(x)=e^{-\beta\,d(x)}. $$

**Air light is added.** Ambient light (strength $A$ per unit scattering) is scattered into the line of sight all along the path; the light added over distance $d$ is

$$ \int_0^{d}A\,\beta\,e^{-\beta z}\,\mathrm{d}z=A\,(1-e^{-\beta d}). $$

Adding the two parts gives the **atmospheric scattering model** (Koschmieder; Narasimhan and Nayar):

$$ \boxed{I(x)=J(x)\,t(x)+A\,\big(1-t(x)\big),\qquad t(x)=e^{-\beta d(x)}.} $$

The veil $V(x)=A(1-t(x))$ is **zero at distance 0 and grows to $A$ at infinity**. Our code uses $\tau=\beta\,d_{\text{far}}$, the optical thickness at the farthest point, as the single "how thick" number.

**Contrast.** Over a patch where $t$ is nearly constant, the local standard deviation scales: $\sigma_I=t\,\sigma_J$. Haze therefore lowers local contrast by $t$ and lifts brightness by $V$.

**The dark channel prior** (He, Sun and Tang). Define $\operatorname{dark}(I)(x)=\min_{y\in\Omega(x)}\min_c I_c(y)$ over a patch $\Omega$. Dividing the model by $A_c$ and taking the patch minimum, with $t$ constant on the patch:

$$ \min_{\Omega}\min_c\frac{I_c}{A_c}=t\,\min_{\Omega}\min_c\frac{J_c}{A_c}+(1-t). $$

The prior says the first minimum on the right is about 0 for clear outdoor pictures, so

$$ \tilde t(x)=1-\omega\,\min_{\Omega}\min_c\frac{I_c(y)}{A_c},\qquad J=\frac{I-A}{\max(\tilde t,t_0)}+A. $$

$\omega<1$ keeps a little haze; $t_0$ stops the division blowing up.

**Our measurement of the prior** (F31): the dark channel of the clear pictures averages 0.12 to 0.21, not 0, with 41 to 62% of pixels below 0.1. The prior holds *moderately*, so DCP is an approximation. **Where it fails** (F32, F33): bright white surfaces look like haze, and for media whose light is not bright (water) the airlight estimate is badly wrong (0.50 too bright).

### Frequency domain

*Constant $t$ and $A$ (a uniform fog):*

$$ \hat I(f)=t\,\hat J(f)\quad(f\neq 0),\qquad \hat I(0)=t\,\hat J(0)+A(1-t)\,N. $$

**Haze scales every non-zero frequency by the same factor $t$ and only moves the average.** So $R(f)=t$ at every $f>0$. *Measured:* with a uniform fog our synthetic pictures give $R=0.301$ against the predicted $e^{-1.2\cdot2\cdot0.5}=0.301$ at every frequency, error 0.0 (F42).

*Distance-dependent $t(x)$ (real haze):* multiplication in space is convolution in frequency,

$$ \hat I=\hat T\circledast\hat J+A\,(\hat 1-\hat T). $$

The veil $A(1-t)$ varies only as fast as the depth does, so it lives at the **lowest frequencies**. The scene's detail is scaled by the **local** $t$, and since a picture's power sits where it has detail, the ratio is a detail-weighted average of $t$, not the plain average:

$$ R(f)\approx\sqrt{\frac{\sum_x t(x)^2\,e(x)}{\sum_x e(x)}},\qquad e(x)=|\nabla J(x)|^2. $$

*Measured:* the plain average predicted 0.47, the measurement was 0.561; the detail-weighted formula predicts 0.553 and cuts the error from 0.103 to 0.029 (F43).

**Real haze agrees with the theory.** The mid-frequency spectral ratio of each of the 185 real pairs correlates **+0.79** with the transmission found by an independent spatial fit (F42).

**Inverting haze in the frequency domain.** The ideal restoration multiplies every non-zero frequency by $1/t$ and sets the average right. Noise is amplified by the same $1/t$, which is why a floor $t_0$ is needed. With white noise of power $S_{nn}$ and picture power $S_{JJ}$, the best linear filter at a frequency (Wiener) is

$$ H(f)=\frac{t\,S_{JJ}(f)}{t^{2}S_{JJ}(f)+S_{nn}}, $$

close to $1/t$ where the picture dominates the noise and rolling off where it does not.

### Haze removal by subtraction (derived, then tested)

Blurring the model: $G*I\approx t\,(G*J)+V$ with $G$ a very wide Gaussian. Our **prior** is that a clear picture's blurred version is about its mean brightness $m$ (we measured $m=0.378$), so the veil is

$$ \hat V(x)=(G*I)(x)-\hat t\,m. $$

The transmission follows from the detail strength in a mid-frequency band (the band power scales by $t^2$): $\hat t=\sigma_{\text{band}}(I)/\sigma_{\text{band}}^{\text{clear}}$ with $\sigma^{\text{clear}}_{\text{band}}=0.0854$ measured on clear pictures. Then $J\approx(I-\hat V)/\hat t$. **Results on real held-out haze pairs** (F45, F46): a *constant* veil gives 11.4 dB (worse than the 12.2 dB of doing nothing); the *local* veil above gives **15.04 dB** (SSIM 0.634), against guided DCP 13.85 dB; the best possible isotropic linear frequency filter (learned from half of the pairs, an upper bound) gives 15.43 dB. Lesson: *a constant veil is wrong because the veil depends on depth; it has to be local.*

### Calibration to real haze (F36, F37, F39)

| | Original synthetic | Fitted to real |
|---|---|---|
| Airlight $A$ | 0.92 | **0.60 to 0.70** |
| Optical thickness $\tau$ | 1.0 / 2.4 / 4.8 | O-HAZE 1.7, I-HAZE 2.1, NH-Haze 3.2, Dense-Haze 8.0 |
| Fog density $\beta$ (scene depth 2.0, by signature match) | 0.5 / 1.2 / 2.4 | 1.3 / 2.2 / 5.5 (mild / medium / heavy) |

---

## 2. Underwater (turbid water)

### Spatial domain

Water absorbs and scatters each colour at its own rate, and **red goes first**. Per colour $c$, the simple model is the haze model with its own coefficient:

$$ I_c=J_c\,e^{-\beta_c d}+B_c\,\big(1-e^{-\beta_c d}\big),\qquad \beta_R>\beta_G>\beta_B, $$

with $B_c$ the colour of the water light (blue-green). This is what `degrade/water.py` implements ($\beta=0.9,\,0.35,\,0.15$; $B=0.10,\,0.42,\,0.50$).

**Literature check** (Akkaynak and Treibitz, CVPR 2018 and 2019): this atmospheric form is only an approximation underwater. In their revised model the **backscatter** and the **direct signal** do not share a coefficient, and the signal's attenuation depends on range and reflectance:

$$ I_c=J_c\,e^{-\beta_c^{D}\,d}+B_c^{\infty}\big(1-e^{-\beta_c^{B}\,d}\big),\qquad \beta_c^{D}\ne\beta_c^{B}. $$

(This form is quoted from memory of the papers; the search snippets did not show the equation, so check the notation before citing.) Our model sets $\beta^{D}=\beta^{B}$, a simplification.

**Colour cast.** Taking the log of the colour ratio of the direct light, $\log(I_R/I_G)\to-(\beta_R-\beta_G)\,d$ at large $d$: the colour cast grows linearly with distance. Our measure `colour_cast` $=\log\big(R/\operatorname{mean}(G,B)\big)$ follows this.

### Frequency domain

For a uniform depth the model is linear and scales every non-zero frequency of each channel by its own transmission: $R_c(f)=t_c=e^{-\beta_c d}$, with the veil in the zero frequency only. *Measured:* with $d_{\text{far}}\times 0.5$, the three channels give 0.259, 0.592 and 0.799, exactly the predicted values (error 0.0, F42). With real depth the same detail-weighting as for haze applies per colour.

**Why DCP and UDCP fail here** (F32): they estimate the water light from the brightest scene pixels, which are bright objects (about 0.84), while the true water light is a dark blue-green (mean 0.34). With $A$ wrong by 0.50, the inversion $J=(I-A)/t+A$ is wrong at every pixel. **Colour fixes** (`red_comp`, `von_kries`) do not need $A$, only the per-channel ratios, which is why they win on the synthetic water (but see the caveat that the synthetic water follows the model they undo).

---

## 3. Glass (film, smudges and droplets)

Three effects with different physics; our model applies them in this order.

### 3a. Milky film (fog or dust on the pane)

$$ I(x)=J(x)\,t_v(x)+A_g\big(1-t_v(x)\big),\qquad t_v(x)=1-s\big(0.4+0.6\,n(x)\big), $$

with $n(x)\in[0,1]$ a smooth noise field. This is the haze model with **no dependence on distance** (the film is on the glass) and a $t_v$ that varies slowly over the picture. *Frequency domain:* the detail is scaled by $t_v$ and the veil $A_g(1-t_v)$ lies at low frequencies, as for haze. *Inversion:* a linear stretch undoes a constant $t_v$ exactly; flat-field correction handles a slow variation. This is why a plain `stretch` wins on the synthetic film (F34) and why it **cannot** be expected to work on raindrops.

### 3b. Smudges

$$ I=(1-\alpha(x))\,I_{\text{film}}+\alpha(x)\big(0.8\,(g_s*I_{\text{film}})+0.2A_g\big), $$

a blend, with strength $\alpha$, between the picture and a blurred, dimmed copy of it.

### 3c. Droplets (rain on glass)

**Literature check** (Qian et al., CVPR 2018): rain on glass is $I=(1-M)\odot B+R$, a background $B$ seen where there is no drop and a drop layer $R$ where the mask $M$ is 1 (a soft version replaces $M$ by a transparency matrix). Our model is that structure with $R=0.9\,(g*J)+0.1A$, a blurred and slightly darkened copy of the scene (the drop acts as a small lens), and a soft mask $w=s\cdot\text{mask}$:

$$ I=(1-w)\,J+w\,\big(0.9\,(g*J)+0.1A\big). $$

*Frequency domain.* Let $D=J-0.9\,(g*J)-0.1A$ be what the drop removes, with $\hat D(f)=\hat J(f)\,h(f)$, $h(f)=1-0.9\,\hat g(f)$ (a Gaussian blur gives $\hat g=e^{-2\pi^2\sigma^2f^2}$). Then $\hat I=\hat J-\hat w\circledast\hat D$. Split the mask into its mean $\mu$ and fluctuation: the **coherent** part scales the picture by $1-\mu h(f)$, and the fluctuation scatters energy between frequencies (an **incoherent** part, growing with the variance of $w$):

$$ R(f)\approx\sqrt{\big(1-\mu\,h(f)\big)^{2}+\operatorname{Var}(w)\,h(f)^{2}}. $$

*Measured on our synthetic drops:* the coherent term alone predicted 0.727 against 0.913 measured; adding the variance term gives 0.812 (error 0.176 to 0.097; the remaining gap is because the drop pattern is not white). *Measured on real raindrops* (58 pairs, F44): $R\approx0.90$, nearly flat; the formula fitted to it gives mean weight $\mu=0.18$ and blur $\sigma=0.8$ px. **Real drops barely blur the picture and barely change brightness or the dark channel; they replace small patches.** That is why the veil model was a poor match (F38) and the **droplets-only model** (300 droplets, strength 0.8) matches the signature (F39) and predicts the real method rankings with Spearman +0.99 on the same scenes (F41).

---

## 4. Translucent or aged plastic

### Spatial domain

A translucent sheet **attenuates**, **tints**, **blurs** (it scatters light by a small angle) and adds a **milky veil**:

$$ I=t\,\big(\kappa\odot(h*J)\big)+A\,(1-t), $$

with $t$ a single number, $\kappa$ the per-colour tint (yellowed plastic loses blue) and $h$ a Gaussian blur. **Literature check:** under-display-camera papers (Feng et al. CVPR 2021; Zhou et al. 2020) model exactly these terms for a display panel in front of a lens: a transmission factor, a wavelength-dependent absorption, a shift-invariant PSF with a long diffraction tail, and veiling flare, noting that sensor saturation breaks the single-PSF assumption.

### Frequency domain

The convolution becomes a product, so the picture is multiplied by a Gaussian low-pass:

$$ \hat I(f)=t\,\kappa\,e^{-2\pi^2\sigma^2f^2}\,\hat J(f)\quad(f\ne0). $$

*Measured:* the synthetic pictures match $R(f)=t\,\bar\kappa\,e^{-2\pi^2\sigma^2f^2}$ with error 0.0007 (F42). **Restoration** is the Wiener deblur $\hat J=\dfrac{H^{*}}{|H|^{2}+K}\hat I$ with $H=e^{-2\pi^2\sigma^2f^2}$, after removing the veil and dividing by $t\kappa$; it works only at mild blur because the Gaussian $H$ is tiny at high $f$ (noise amplification). We found no real plastic data yet (open item).

---

## 5. Stacked media

Apply two layers in order, scene-side first. With $I_1=t_1J+A_1(1-t_1)$ and $I_2=t_2I_1+A_2(1-t_2)$:

$$ \boxed{I_2=t_1t_2\,J+t_2A_1(1-t_1)+A_2(1-t_2)}, $$

the proposal's stacked equation. For $n$ layers, $t_{\text{eff}}=\prod_i t_i$ and the total added light is $\sum_iA_i(1-t_i)\prod_{j>i}t_j$: **light added early is attenuated by every later layer**.

*Frequency domain.* For layers that act linearly and shift-invariantly, the effect on a picture's spectrum **multiplies**: $R_{\text{stack}}(f)=R_1(f)\,R_2(f)\cdots$. *Measured* (water behind plastic): measured and predicted agree to an error below 0.0004 for all three colours (F42).

**Order of restoration.** The camera-side layer acted last, so its affine map must be undone first: $J=\big((I_2-A_2(1-t_2))/t_2-A_1(1-t_1)\big)/t_1$. In practice (F34, F35) this helped for water stacks (+0.6 to +2.6 dB) but not when the inner layer is haze and the outer is plastic (-1.1 dB), because each undoing step is an *estimate* that relies on assumptions (DCP's dark channel) that the other layer breaks.

---

## 6. Summary of the equations, and what changed in the code because of this

| Medium | Spatial equation | Frequency-domain statement | Checked by measurement | What was updated |
|---|---|---|---|---|
| Haze | $I=Jt+A(1-t)$, $t=e^{-\beta d}$ | $R(f)=t$; veil at low $f$; with depth, detail-weighted $t$ | exact for uniform fog; real pairs correlation +0.79; depth formula error 0.103 to 0.029 | calibrated presets: $A\approx0.6$ to $0.7$, $\beta=1.3/2.2/5.5$ |
| Water | $I_c=J_ce^{-\beta_cd}+B_c(1-e^{-\beta_cd})$ | $R_c(f)=t_c$ | exact for three colours | revised model (separate backscatter coefficient) added as an option |
| Glass film | haze with $t_v(x)$ slowly varying, no depth | like haze, veil at low $f$ | by the haze checks | none |
| Glass droplets | $I=(1-w)J+w(0.9\,g*J+0.1A)$ | $R=\sqrt{(1-\mu h)^2+\mathrm{Var}(w)h^2}$ | real drops $R\approx0.90$, fitted $\mu=0.18$; error 0.176 to 0.097 | **droplets-only** calibrated preset (veil and smudge off) |
| Plastic | $I=t\kappa(h*J)+A(1-t)$ | $R=t\kappa\,e^{-2\pi^2\sigma^2f^2}$ | error 0.0007 | none |
| Stack | $I_2=t_1t_2J+t_2A_1(1-t_1)+A_2(1-t_2)$ | $R=\prod R_i$ | error below 0.0004 | none |

> [!note] What the theory explains about the results
> - **Why DCP wins on haze but fails on water and plastic:** its prior and its bright-airlight assumption match haze only.
> - **Why a contrast stretch wins on synthetic glass and plastic:** both are affine ($Jt+\text{constant}$), and a stretch is the exact inverse of an affine map. It cannot work on raindrops, which are a masked blend.
> - **Why a constant-veil subtraction fails and a local one works:** the veil is $A(1-t(x))$ and $t$ depends on distance.
> - **Why no linear frequency filter can do much better than about +3 dB on real haze:** with $t(x)$ varying, the haze is not a single linear space-invariant operation, so the best such filter (15.4 dB) is a ceiling.

Back to [[Subjects/DIP/Projects/DIP-Project/DIP-Project|DIP-Project]] · [[Subjects/DIP/Projects/DIP-Project/docs/results-analysis|results-analysis]] · [[Subjects/DIP/Projects/DIP-Project/log|log]]
