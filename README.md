# KRW Trendanalyse Grondwaterkwantiteit

De Python package `krw_trendanalyse` bevat functionaliteit om trend analyse uit
te voeren op stijghoogtereeksen voor de Kader Richtlijn Water (KRW).

De trend analyse kan uitgevoerd worden op individuele tijdreeksen of op de
residuen van tijdreeksmodellen. De resultaten kunnen geaggregeerd worden per
grondwaterlichaam. De package biedt ook mogelijkheden om de resultaten te
visualiseren.

## Installatie

Installeren kan met:

`pip install krw_trendanalyse`


## Gebruik

Een kort voorbeeld van hoe de package toegepast kan worden om te onderzoeken
of er een trend aanwezig is op basis van de gemiddeldes per periode.

```python
import krw_trendanalyse as krw
import pandas as pd

series  = pd.read_csv(...) # tijdreeks stijghoogte

# definieer periodes (referentie periode en 1+ periodes om te vergelijken)
periods = [
    (2000, 2005),
    (2018, 2023),
]

# bereken gemiddeldes inclusief onzekerheid per periode
trend_mean = krw.mean_per_period(series, periods)
```

## Auteurs

De implementatie in Python is geschreven door:

- Davíd Brakenhoff, Artesia, 2025

De methode van trendanalyse is in het verleden ontwikkeld door o.a.

- J. van Asmuth
- ...

## Referenties
