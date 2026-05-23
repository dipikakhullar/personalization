# User Traces — r/AskStatistics

*Source: `../data/extracted_current/pairs/sub-AskStatistics.jsonl`*

- pairs: 12,811
- unique users: 7,586
- pairs with both signals attached: 12,811 (100%)
- top_comment == preferred_answer: 5,571 (43%)
- divergent: 7,240 (56%)

Three users with distinct profiles are shown below. Interactions are sorted ascending by timestamp; picks are on distinct posts so the timeline reflects actual different threads.

## User A — heavy user (most interactions)

**`anon_3e27f193d61ee420`** — 52 interactions across 20 distinct posts, 2018-12-20 → 2024-09-23, 35/52 divergent (67%).

Highest interaction count in this subreddit. Read top-to-bottom for a snapshot of how this user engages over time — what they ask, what answer style they thank, whether their preferred answer matches the community top vote.

### Interaction 1 of 52 — 2018-12-20

> **Q:** How does sample size mitigate Simpson's paradox? Good day all, I have a question. I am a newbie to stats, so please excuse my lack of knowledge. I came across this article, which explains Simpsons paradox…

- **preferred_answer** *(score = 6)*:
  > "*Simpson's Paradox arises when there are very different effect sizes in different subgroups with very different sample sizes in those subgroups. In the example they give of Android vs IOS conversions, it arises because phone conversion rates are much lower than for tablets and Android is mostly phones whereas IOS is mostly tablets. If you don't take account of phone vs tablet it makes the Android conversion rate look lower than IOS even though it…*"
- **top_comment:** same as preferred (signals agree)
- **OP's thanks-reply:** "*Thanks D-juice for your help. I feel like your answer confused me more. The more I delve into this sort of thing the more I realise I am out of my depth. I am a marketer with huge interest in a/b testing and conversion optimisation. How long do you…*"

### Interaction 13 of 52 — 2020-05-25

> **Q:** What is the correct kind of test to use if I want to know if respondents agree/disagree with a statement before and after exposure to a stimulus? For example: If I have a sample of 100 people. I want to ask them the a question like, do you agree or disagree with this statement. Then expose them to a video, and then ask them after the video if they…

- **preferred_answer** *(score = 3)*:
  > "*McNemar's test for the significance of change.*"
- **top_comment:** same as preferred (signals agree)
- **OP's thanks-reply:** "*Thank you very much. Would McNemar also be suitable if I asked the participant to fill in a likert scale question?*"

### Interaction 25 of 52 — 2021-05-14

> **Q:** How do I deal with SDs that go into negative integers for things like price? I mean, if I am looking at the SD of my income from my website for instance. What if the SD take me to -$34 at the lower end. No one is paying -$34 for anything.

- **preferred_answer** *(score = 2)*:
  > "*I'm not sure I fully understand your example, but usually Normal distributions are a useful generalization for the distributions of variables that must always be above zero. In the instance where the boundary is strict at zero, you can instead use a Rayleigh distribution. In general price does not have a strict boundary at zero—a negative price simply means that someone would pay you to take the product off their hands. This happened briefly…*"
- **top_comment** *(score = 7)*:
  > "*Standard deviation is just a measure of variation; and while it is convenient to interpret it as "most of the time my data will be within +/- one standard deviation from the mean", this is certainly not always true, as you are finding out. Most data involving $$$ are very right skewed, and you have to just understand that with this kind of data the standard deviation is made much higher by those few high numbers, and the +/- one standard…*"
- **OP's thanks-reply:** "*Thank you for the help. So lets say for example I am looking at my average order value (AOV) over the last year for 3 cohorts of users. I want to look at which cohort has the most variation. If my AOV is lets say for one cohort is $40 and my SD is…*"

### Interaction 39 of 52 — 2021-09-26

> **Q:** How would you intuitively interpret the correlation of these 2 plots?

- **preferred_answer** *(score = 51)*:
  > "*No correlation between x & y for either plot. The top plot has a wider range of y values but neither plot indicates that y values are affected by x values.*"
- **top_comment** *(score = 136)*:
  > "*Both are uncorrelated / almost uncorrelated The main difference is the choice of scale on the y-axis. If you get rid of the wasted white space on the second scale they'll look very similar*"
- **OP's thanks-reply:** "*This what I thought, but my brain was doubting itself. Thanks*"

### Interaction 52 of 52 — 2024-09-23

> **Q:** Is this a correct statement "Sample size in a datasets can give your inaccurate results when comparing medians"? Hey everyone, I have a question that might seem a bit simple, but I could use some clarification. I’m comparing the medians of two datasets. One of these datasets has nearly twice as many rows (or samples) as the other. My concern is…

- **preferred_answer** *(score = 6)*:
  > "*If the data from both groups is randomly sampled, why would you expect the larger dataset to have more values on the lower end? If a random sample showed data as the one you simulate here, you would have reason to believe the central tendency in population A is lower than in population B, sample size would not influence that.*"
- **top_comment:** same as preferred (signals agree)
- **OP's thanks-reply:** "*Thanks for the guidance. So, col A would be what I am using to determine median performance on my social media campaign (all data over 3 months. And col B would be my campaign performance. So what I am trying to understand is did my campaign (B) do…*"

## User B — longest temporal span

**`anon_5ec7072440cabd46`** — 4 interactions across 4 distinct posts, 2013-09-27 → 2024-03-05, 0/4 divergent (0%).

The user whose interactions are spread over the most years. Interesting for seeing how question topics evolve while writing voice and answer-style preferences stay stable.

### Interaction 1 of 4 — 2013-09-27

> **Q:** How do I make this simple table in SAS? I have two variables, Question1 and Demo1. They look like this: Question1 * 3 - Agree * 2 - Neutral * 1 - Disagree Demo1 * 1 - Young * 2 - Old I want a table like this. Row percentages and then a total n count at the end for each demographic. Labels|Agree|Neutral|Disagree|Total Count :--|:--|:--|:--|:--…

- **preferred_answer** *(score = 1)*:
  > "*Try: proc freq data=mydata; tables Demo1 * Question1 /nocol noper nofreq ; run; For the second part, with a lot of fiddling with ods output. It could be done if you're desperate, but if you're doing it once, do it by hand.*"
- **top_comment:** same as preferred (signals agree)
- **OP's thanks-reply:** "*Thanks for the response!*"

### Interaction 2 of 4 — 2016-06-02

> **Q:** How to create bayesian credibility intervals similar to confidence intervals from a sample? Preferably an example in R or SAS. Most of the text I've read jump straight to significance testing. I just want to know estimates from my sample. Also, does the coding change if it is an ordinal variable, continuous variable. I think I got the binary…

- **preferred_answer** *(score = 5)*:
  > "*Here's an R example set.seed(2) # generate data theta.true <- 0.33 n <- 100 k <- sum(rbinom(n, 1, theta.true)) # 95% credible interval (quantile based) # flat prior (Beta(1, 1)) qbeta(c(0.025, 0.975), 1 + k, 1 + n - k) # [1] 0.2189787 0.3961471 # very informative prior (Beta(333, 667)) qbeta(c(0.025, 0.975), 333 + k, 667 + n - k) # [1] 0.3060779 0.3617678 # for comparison, frequentist 95% confidence interval prop.test(k, n)$conf.int # [1]…*"
- **top_comment:** same as preferred (signals agree)
- **OP's thanks-reply:** "*Thanks! I appreciate the example. This is good, I am actually interested in understanding the differences of bayesian vs. frequentist approach and where they diverge.*"

### Interaction 3 of 4 — 2018-03-29

> **Q:** Why is a sig test of sub-group vs. non-subgroup the same as sub-group vs. entire group(including subgroup)? I've run into a weird work problem where the client has asked us to make subgroup comparisons with the entire group e.g., A vs. A + B + C My initial reaction was that is not right, you should rather do: A vs. B + C But they responded to show…

- **preferred_answer** *(score = 1)*:
  > "*It is only the same if you account for the dependence in the A vs. All situation, otherwise they are not the same. In other words, you can do the analysis either way, but they require different statistical tests. The conversation you link to is a little obscure, but both you and your client are correct. You could do it either way.…*"
- **top_comment:** same as preferred (signals agree)
- **OP's thanks-reply:** "*Thank you for the clarification!!! So is there a formalize off the shelf test to "fix" the clients proposed approach? Or is essential running the test as, A vs. B + C, the solution.*"

### Interaction 4 of 4 — 2024-03-05

> **Q:** Designing an experiment that only measures main effects but minimizes trials I've been reading a little bit about fractional factorial designs, and they all seem to be interested in preserving some ability to measure interaction terms. However, if I know that interaction terms are not possible a priori, is there a way to design an experiment that…

- **preferred_answer** *(score = 1)*:
  > "*> is there a way to design an experiment that minimizes the number of trials needed to explore just the main effects? sure, there's a number of ways to go about it depending on the specific details. are these all binary factors? You need to make sure its balanced on each factor, and you want to try (for various reasons) to keep them balanced (orthogonal) across factors, at least pairwise. If fractional factorial designs that just permit all…*"
- **top_comment:** same as preferred (signals agree)
- **OP's thanks-reply:** "*Thanks for your response. In reading your response, I think I answered my own question. Would the Plackett–Burman Design be the optimal design for binary factors and optimizing for minimum number of trials for main effects? (Assume trials are costly…*"

## User C — strongest individual signal (highest divergence)

**`anon_49e92e5e70eef6b7`** — 8 interactions across 7 distinct posts, 2017-01-31 → 2018-10-21, 7/8 divergent (87%).

Among users with enough activity (>=8 interactions, >=5 distinct posts), this one most consistently thanks an answer that wasn't the community-top — i.e. their personal preference diverges from the population most strongly. Useful as a personalization difficulty case.

### Interaction 1 of 8 — 2017-01-31

> **Q:** How can your overall ANOVA deliver an insignificant p value, but your effect tests deliver the opposite? (JMP, 2x4 ANOVA) Hi, it's been a while since I've taken a stats class, so sorry if this is remedial: I have an experiment I designed to be a 2x4 factorial ANOVA (2 species, 4 treatment types labeled "tubes"). I have a variety of response…

- **preferred_answer** *(score = 1)*:
  > "*No, you can, I guess. I don't take too much look at the model F.*"
- **top_comment** *(score = 2)*:
  > "*In your model you have 7df. It shouldn't be surprising that the average SSQ of these 7 is lower than the highest sum of squares (Sp) in your design. I wouldn't have considered the test of the model as a whole. What was your purpose in considering it (other than it was part of the output)? If it was to control the Type I error rate you might have been better off using some type of Bonferroni correction.*"
- **OP's thanks-reply:** "*Ok, thanks for your time!*"

### Interaction 4 of 8 — 2017-03-03

> **Q:** What to do with Cauchy distribution of residuals How do you analyze data that has (what looks to me) like cauchy distribution? It is not normally distributed and has unequal variances. I've been told I cannot do ANOVA with this data. **Edit, messed up the formula:** The data are %growth (final-initial)/initial But some are negative (they shrunk,…

- **preferred_answer** *(score = 2)*:
  > "*If you took logs and fit the normal you would normally use the identity link (that should be similar to the gamma on untransformed ratios with log link) > Do I just use AIC to decide which one to use? I wouldn't. I'd lean toward using the gamma (because once you start accounting for variance-parameter uncertainty, mean predictions and CIs on lognormal models have a problem that's difficult to avoid). So if gamma works for you, stick to that.*"
- **top_comment** *(score = 2)*:
  > "*> has unequal variances The Cauchy distributions all have infinite variances. I bet you any amount you like that don't actually have Cauchy distributions (and I bet I can prove it) > %growth (final/initial)/intial That's not % growth. % growth would be (final-initial)/initial ... which is the same thing as (final/initial) - 1. Don't subtract the 1! Try looking at log(final/initial); that's a common thing to look at as long as your final and…*"
- **OP's thanks-reply:** "*You're awesome, thanks so much for your help. My stats knowledge is outdated and my current advisor isn't too concerned with how they turn out. The raw data give blatant evidence for significance, but without having the stats to back it up I felt…*"

### Interaction 6 of 8 — 2018-04-09

> **Q:** Can someone tell me if these residuals are ok? If not why? https://mokacytle.tumblr.com/post/172760751322 I'm trying to visualize my data and it's kind of messy. I was thinking maybe I need to transform the data, but the residuals from this ANCOVA are normally distributed, so I'm not sure if I should do anything to them. If I have a really low R…

- **preferred_answer** *(score = 3)*:
  > "*> the residuals from this ANCOVA are normally distributed No they aren't. (I can say this with absolute certainty in this case) Presumably you mean you checked normality (likely failed to reject normality in a test). That failure doesn't mean you can assert normality. I can probably identify dozens of non-normal distributions you also wouldn't reject, but it cannot possibly be more than one of them (and would probably actually be none of them).…*"
- **top_comment** *(score = 3)*:
  > "*mh. One problem I see with the percental approach is that you control for initial size in a sense. The model does not know how big the initial size was and therefore a lot of uncertainty is introduced. If I where to ask you how likely it is that plant a will double in size after a few days, your answer would largely depend on the initial size of the plant .It is very easy for a small plant to double in size but hard for a big plant. The model…*"
- **OP's thanks-reply:** "*Thanks for your reply! You’ve hit on a question I forgot! The measure for water flow (the x) does have multiple measures of growth (the y) because each cage had four individuals of each species but only one device measuring flow. Do you have a…*"

### Interaction 8 of 8 — 2018-10-21

> **Q:** What to do with non-normal distributions of continuous variable residuals (even after transformed) I counted densities of two mussel species along a river, and I'd like to see whether there is a correlation between river mile and density. The basic trend is that one species decreases over the length of the river, but the other species increases.…

- **preferred_answer** *(score = 3)*:
  > "*>the residuals are not normally distributed, even when the raw data are log or square root transformed This sort of implies that you just did the transformations blind rather that based on what the data look like? Did you use scatter plots to consider whether you needed to transform and if so, what transformation(s) to use? This is a useful quick guide on how to approach it: [Transformations to Improve Fit and Equalize…*"
- **top_comment:** same as preferred (signals agree)
- **OP's thanks-reply:** "*Thanks for this resource. And yes, I’m guilty. I log transformed originally to try to limit outliers and bc I’m trying to update two papers in the literature that log transformed their data. I tried sqrt because I know it’s supposed to be better for…*"
