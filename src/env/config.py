from dataclasses import dataclass, field

@dataclass
class SpeciesConfig:
    num: int = 1
    speed: float = 1.0

@dataclass
class PredatorConfig:
    num: int = 2
    speed: float = 1.5

@dataclass
class FeedConfig:
    num: int = 4

@dataclass
class EnvConfig:
    env_size: int = 128
    species: SpeciesConfig = field(default_factory=SpeciesConfig)
    predator: PredatorConfig = field(default_factory=PredatorConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)

