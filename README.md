# InterpretableTS

### Rudi

Get stats:

```
python extract_stats.py --prefix gru --folder data/stock100
```

Get rules from stats:

```
python extract_rules.py --prefix gru --folder data/stock100 --device cuda:0
```

Get features generated from rules:

```
cd rudi
python features.py
```

Train MLP with generated features:

```
python main.py
```

