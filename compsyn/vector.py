from __future__ import annotations

from pathlib import Path

from .trial import Trial, get_trial_from_env
from .s3 import upload_file_to_s3, download_file_from_s3, s3_object_exists


class VectorNotGeneratedError(Exception):
    pass


class MissingRevisionNameError(Exception):
    pass


class BadPickleError(Exception):
    pass


def load_pickle(filename: Union[str, Path]) -> Any:
    """
    load a saved pickle
    """
    with open(filename, "rb") as f:
        obj = pickle.load(f)
        return pickle.load(f)


# Abstract Base Class
class Vector:
    """
    A Vector may hold embeddings generated from any sensory input.
    A Vector may hold links to data used to generate the Vector as well as analysis data in a Backend object.
    The Vector class can be associated with a Trial object, which can be used to logically organize results.
    An optional Backend will be included to facilitate collaboration.

    The Vector object will implement some of the functionality, but also serves as an abstract base class for Vector subclass implementers.
    """

    def __init__(
        self, label: str, revision: Optional[str] = None, trial: Optional[Trial] = None
    ) -> None:
        #: label for the vector
        self.label = label
        #: optional revision string to use for saving to a shared backend
        self.revision = revision
        #: metadata to associate with results can be configured through a Trial dataclass
        if trial is None:
            # default to getting trial metadata from the environment
            self.trial = get_trial_from_env()
        else:
            self.trial = trial
        #: track whether the information the vector represents is locally available as attributes
        self._attributes_available: bool = False
        #: local path for saving and loading from a pickle
        self._local_pickle_path: Path = self.trial.work_dir.joinpath(
            self.vector_pickle_path
        )

    @property
    def vector_pickle_path(self) -> Path:
        if self.revision is None:
            raise MissingRevisionNameError(
                f"set {self.__class__.__name__}.revision before pushing"
            )
        return (
            Path(f"{self.trial.experiment_name}/wordtocolorvectors")
            .joinpath(self.revision)
            .joinpath(self.word)
            .joinpath("w2cv.pickle")
        )

    @abstractmethod
    def create_embedding(self, **kwargs) -> Vector:
        # run analysis from folder of local data, generating a Vector
        pass

    def save(self):
        """
        save a Vector as a pickle
        """
        with open(self._local_pickle_path, "wb") as f:
            pickle.dump(self, f)

    def pull(self, overwrite: bool, **kwargs) -> None:
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
        obj = load_pickle(self._local_pickle_path)
        if not isinstance(obj, self.__class__):
            raise BadPickleError(f"{obj} loaded from pickle is not a {self.__class__}.")
        # bad idea to replace self like this?
        self.__dict__.update(obj.__dict__)

    def push(self, overwrite: bool, **kwargs) -> None:
        """
        Optional remote Backend integration point
        Subclasses of Vector should call super().push(**kwargs) if they extend push
        """
        self.save()
        upload_file_to_s3(
            local_path=self._local_pickle_path, s3_path=self.vector_pickle_path,
        )
