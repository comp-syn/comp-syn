from __future__ import annotations

import pickle
from pathlib import Path

from .logger import get_logger
from .trial import Trial, get_trial
from .s3 import (
    upload_file_to_s3,
    download_file_from_s3,
    s3_object_exists,
    NoObjectInS3Error,
)
from .utils import human_bytes


class VectorNotGeneratedError(Exception):
    pass


class MissingRevisionNameError(Exception):
    pass


class BadPickleError(Exception):
    pass


def load_vector_pickle(filename: Union[str, Path]) -> Any:
    """
    load a saved pickle
    """
    with open(filename, "rb") as f:
        obj = pickle.load(f)
        get_logger("load_pickle").info(f"loaded pickle from {filename}")
        return obj


# Abstract Base Class
class Vector:
    """
    A Vector may hold embeddings generated from any sensory input.
    A Vector may hold links to data used to generate the Vector as well as analysis data in a remote backend.
    The Vector class is associated with a Trial object, which can be used to logically organize results.

    The Vector class will implement some of the shared functionality for persistence, but implements no logic for gathering data or creating "vector" analysis data.
    """

    def __init__(
        self,
        label: str,
        revision: Optional[str] = None,
        trial: Optional[Trial] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        #: label for the vector
        self.label = label
        #: optional revision string to use for saving to a shared backend
        if revision is None:
            self.revision = "unnamed"
        else:
            self.revision = revision
        #: experiment metadata to associate with results can be configured through a Trial dataclass
        if trial is None:
            # default to getting trial metadata from the environment
            self.trial = get_trial()
        else:
            self.trial = trial

        #: other metadata can also be tracked by directly passing it
        self.metadata = metadata
        #: track whether the information the vector represents is locally available as attributes
        self._attributes_available: bool = False

    def __repr__(self) -> str:
        output = f"{self.__class__.__name__}({self.label})\n\t"
        output += "\n\t\t".join(str(self.trial).split("\n\t"))
        if self.metadata is not None:
            output += "\n\tmetadata:\n\t\t"
            output += "\n\t\t".join(
                [f"{key:40s} = {val}" for key, val in self.metadata.items()]
            )

        return output

    @property
    def _local_pickle_path(self) -> Path:
        #: local path for saving and loading from a pickle
        return self.trial.work_dir.joinpath(self.vector_pickle_path)

    @property
    def vector_pickle_path(self) -> Path:
        if self.revision is None:
            raise MissingRevisionNameError(
                f"set {self.__class__.__name__}.revision before pushing"
            )
        return (
            Path(f"{self.trial.experiment_name}/vectors")
            .joinpath(self.revision)
            .joinpath(self.label)
            .joinpath("w2cv.pickle")
        )

    def load(self) -> Vector:
        """
        Load and return a vector from a pickle file
        """
        if not self._local_pickle_path.is_file():
            raise FileNotFoundError(self._local_pickle_path)

        obj = load_vector_pickle(self._local_pickle_path)
        if not isinstance(obj, self.__class__):
            raise BadPickleError(
                f"{obj.__class__.__name__} loaded from pickle is not a {self.__class__}."
            )

        # is it a bad idea to replace self like this?
        self.__dict__.update(obj.__dict__)

    def save(self) -> None:
        """
        save a Vector as a pickle
        """
        self._local_pickle_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self._local_pickle_path, "wb") as f:
            pickle.dump(self, f)
        self.log.info(
            f"saved {human_bytes(self._local_pickle_path.stat().st_size)} pickle to {self._local_pickle_path}"
        )

    def pull(self, overwrite: bool = False, **kwargs) -> None:
        """
        Optional remote Backend integration point
        Subclasses of Vector should call super().pull(**kwargs) if they extend pull
        """
        if not s3_object_exists(s3_path=self.vector_pickle_path):
            raise NoObjectInS3Error(
                f"no S3 object exists for this Vector / Trial combination:\n\t{self.label}.{self.revision} {self.trial}"
            )
        download_file_from_s3(
            local_path=self._local_pickle_path, s3_path=self.vector_pickle_path,
        )
        self.load()

    def push(self, overwrite: bool = False, **kwargs) -> None:
        """
        Optional remote Backend integration point
        Subclasses of Vector should call super().push(**kwargs) if they extend push
        """
        self.save()
        upload_file_to_s3(
            local_path=self._local_pickle_path, s3_path=self.vector_pickle_path,
        )

    # @abstractmethod
    def run_analysis(**kwargs) -> None:
        pass
