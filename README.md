# KRW Trendanalyse Grondwaterkwantiteit

De Python package `krw_trendanalyse` bevat functionaliteit om trend analyse uit
te voeren op stijghoogtereeksen voor de Kader Richtlijn Water (KRW).

De trend analyse kan uitgevoerd worden op individuele tijdreeksen of op de
residuen van tijdreeksmodellen. De resultaten kunnen geaggregeerd worden per
grondwaterlichaam. De package biedt ook mogelijkheden om de resultaten te
visualiseren.

## Installatie

De krw_trendanalyse package is momenteel nog niet beschikbaar op PyPi. Installeren kan momenteel met:

```bash
pip install git+https://github.com/ArtesiaWater/krw_trendanalyse.git
```

Wil je eerst lokaal de code klonen (bijvoorbeeld om eraan te werken), dan kun je de repository downloaden als ZIP via de groene <> Code knop rechtsboven op GitHub, of via de command line:
```bash
git clone https://github.com/ArtesiaWater/krw_trendanalyse.git
```
Pak de ZIP uit (indien van toepassing) en navigeer in de terminal naar de map waarin de krw_trendanalyse-directory staat:

```bash
cd <download_map>/krw_trendanalyse
pip install -e . # make sure environment is activated
```
De `-e` tag ("editable mode") is handig voor ontwikkeling: wijzigingen in de code zijn direct beschikbaar zonder herinstallatie.

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
- Ruben Caljé, Artesia, 2025
- Maud Theunissen, Provincie Noord Brabant, 2025

De methode van trendanalyse is in het verleden ontwikkeld door o.a.

- J. van Asmuth
- ...

## Referenties
