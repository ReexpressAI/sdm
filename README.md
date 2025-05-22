# Similarity-Distance-Magnitude Universal Verification

## Overview

| ![Overview figure for an SDM network.](/paper/sdm_network.png) |
|:--|
| **SDM networks** are uncertainty-aware via a robust estimator of *index-conditional calibration*, $` \hat{p}(y \mid \mathbf{x})_{\rm{lower}} `$, over output verification (i.e., binary classification of instruction-following); intrinsically introspectable via depth-matching into a training set ($` \mathcal{D}_{\rm{tr}} `$) and correspondence to comparable points in a held-out calibration set ($` \mathcal{D}_{\rm{ca}} `$) via $` \left\lfloor\tilde{q}\right\rfloor `$, which is a stable mapping and summary of the epistemic uncertainty signals of $` \rm{Similarity} `$, $` \rm{Distance} `$, and $` \rm{Magnitude} `$; and updatable via a fine-tuning process to maximize the proportion of verifiable high-probability generations. Decoding proceeds by generating from the distribution of $` \rm{SDM}(\mathbf{z}_{\rm{neg}}, \mathbf{z}_{\rm{pos}}) `$ up to a control token at the unit-of-analysis of the verification labels. Decoding then continues, or other branching actions are taken, based on $` \hat{p}(y \mid \mathbf{x})_{\rm{lower}} `$.  |

## Paper

A copy of the paper (arXiv v3) is available [here](/paper/sdm.pdf).

## Research Code and Replication Scripts

The code in the `research_code` directory is provided for archival purposes to replicate the experiments of the research paper. See the README in that directory for instructions.

## Applied Example as an MCP Server

Separately, we provide an example of a pre-trained SDM estimator that you can use with existing LLMs to verify their instruction-following abilities. See the [Reexpress MCP Server repo](https://github.com/ReexpressAI/reexpress_mcp_server) for additional details.

## Citation

```
@misc{Schmaltz-2025-SimilarityDistanceMagnitudeUniversalVerification,
      title={Similarity-Distance-Magnitude Universal Verification}, 
      author={Allen Schmaltz},
      year={2025},
      eprint={2502.20167},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.20167}, 
}
```
