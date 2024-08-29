# Lazy Neural Planner - LNP

This is the repository for our publication _A Lazy Approach to Neural Numerical Planning with Control Parameters_.

## Content
We present a lazy, hierarchical approach to tackle the challenge of planning in complex numerical domains, where the effects of actions are influenced by control parameters, and may be described by neural networks.

You can access the paper here (link will be provided later).

## Requirements 
We recommend Anaconda to install all requirements for our repository. The requirements are saved in the ```venv.yml``` file.

For a quick installation run: ```conda env create -f venv.yml```

## Replication
We benchmarked our approach against the only other approach capable of handling NN-enriched Numerical Planning with Control Parameters (N3PCP) Problems, as presented in this [paper](https://link.springer.com/chapter/10.1007/978-3-031-50485-3_33). The corresponding code is publicly available in this [repository](https://github.com/RHeesch/rainer).

Our benchmarks were conducted across four different domains. We made slight modifications to the Zenotravel and Drone domains from the numerical track of [IPC23](https://github.com/ipc2023-numeric/ipc2023-dataset) to integrate control parameters. Additionally, we adapted the [cashpoint domain](https://github.com/Emresav/ECAI16Domains/tree/master) introduced by [Savaş et al.](https://ebooks.iospress.nl/doi/10.3233/978-1-61499-672-9-1185) by omitting time constraints. Lastly, we utilized the [FliPSI-domain](https://github.com/imb-hsu/FliPSi) domain for our analysis.

![image](https://github.com/user-attachments/assets/9630d5b8-099d-4a75-978f-94a9a736787d)


## Citation
When using the code from this paper, please cite:
```
@incollection{lazyheesch24,
 title={A lazy approach to neural numerical planning with control parameters},
 author={Heesch, René and Cimatti, Alessandro and Ehrhardt, Jonas and Diedrich, Alexander and Niggemann, Oliver},
 booktitle={ECAI 2024},
 year={2024},
 publisher={IOS Press}
}
}
```

## License
Licensed under MIT license - cf. LICENSE
