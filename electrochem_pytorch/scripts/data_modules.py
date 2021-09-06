import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset


class UnaugmentedAnalyteDataset(Dataset):
    pass


class UnaugmentedAnalyteDataModule(pl.LightningDataModule):
    """Data Module that returns only true, unaugmented samples.

    By default it scales the dataset to the range (-1, 1) and label encodes
    y to enhance the performance of PyTorch multiclass classifiers.
    """

	def __init__(
            self,
            data_dir='data/',
            batch_size=50,
            seq_length=1002,
            rescaled_min_val=-1,
            rescaled_max_val=1,
            batch_first=True,
            y_encoding='label',
            random_seed=42,
            shuffle=True,
            validation_split=0.2,
            ):
        """Initialize data module.

        Parameters
        ----------
        data_dir : str, optional
            The directory where the data is stored, by default 'data/'
        batch_size : int, optional
            The number of samples returned in each batche, by default 50
        seq_length : int, optional
            Length of each sample, by default 1002
        rescaled_min_val : int, optional
            The global minimum value of the dataset after scaling has been
            applied, by default -1
        rescaled_max_val : int, optional
            The global maximum value of the dataset after scaling has been
            appled, by default 1
        batch_first : bool, optional
            Whether to re-shape the data such that batch_size is the first
            dimension. If True data has shape (batch_size, seq_length,
            features), otherwise its (seq_length, batch_size, features),
            by default True (to have easier compatibility with Keras)
        y_encoding : str, optional {'label', 'ohe', None}
            The encoding to apply to y. Options are label encoding (0, n-1),
            one-hot encoding or no encoding. The default is 'label' as this is
            what PyTorch expects for multilcass classification.
        random_seed : int, optional
            Random state to ensure reproducibility, by default 42
        shuffle : bool, optional
            Whether to shuffle the data or not before training, by default
            True
        validation_split : float, optional
            The percentage of data to set aside for validation, by default 0.2
        """
		super().__init__()
		self.data_dir = Path(data_dir)
		self.batch_size = batch_size
		self.seq_length = seq_length
		# Scaling params
		self.rescaled_min_val = rescaled_min_val
		self.rescaled_max_val = rescaled_max_val
        self.batch_first = batch_first
        self.y_encoding = y_encoding
		self.seed = random_seed
		self.shuffle = shuffle
		self.val_split = validation_split


	def prepare_data(self):
		"""
		Use this method to do things that might write to disk
		or that need to be done only from a single process.
		e.g. downloading and tokenizing data.

		Not needed for this project.
		"""
		pass


	def _load_dfX_dfy(self):
        """Read in all analytes and return df_X and df_y.
        """
        analytes = pd.read_csv(DATA_DIR / 'all_data.csv', index_col=0)

		df_X = df.iloc[:, :-1]
		df_y = df.iloc[:, -1]

		return df_X, df_y


    def _scale_X_y(
            self,
            df_X,
            df_y,
            scaled_min=-1,
            scaled_max=1):
        """Scale each row in df_X to be in the range [scaled_min, scaled_max].

        Note: the min value of df_X is mapped to scaled_min and the max is mapped
			  to scaled_max. Each row is not mapped to [scaled_min, scaled_max]
              independently.

        Parameters
        ----------
        df_X : pd.DataFrame
            DataFrame where each row is a sample and each column a voltage
            index.
        df_y : pd.DataFrame
            DataFrame with one column containing int labels for each sample
        scaled_min : int, optional
            The value df.min().min() is mapped to. The new global min for the
            dataset as a whole, by default -1
        scaled_max : int, optional
            The value df.max().max() is mapped to. The new global max for the
            dataset as a whole, by default 1

        Returns
        ----------
        df_X_scaled: pd.DataFrame
            DataFrame with each value scaled to sit in the range [scaled_min, scaled_max]
        df_y: pd.DataFrame
            df_y unmodified.
        """
        df_X_scaled = self._scale_df_X_to_range(df_X, scaled_min, scaled_max)
		# No need to scale y
		return df_X_scaled, df_y


	def _scale_df_to_range(
            self,
            df,
            df_global_min=-40,
            df_global_max=40,
            scaled_min=-1,
            scaled_max=1
            ):
        """Scale all rows in df to [scaled_min, scaled_max]. df_global_min is
        mapped to scaled_min and df_global_max is mapped to scaled_max.
        All values inbetween are mapped to their appropriate values.

        Note: This scales the entire dataset row-wise. Each row is not mapped to
              [scaled_min, scaled_max] independently.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame where each row is a sample and each column a feature.
        df_global_min: int, optional
            The global minimum of the DataFrame.
        df_global_max: int, optional
            The global maximum of the DataFrame.
        scaled_min : int, optional
            The value df_global_min is mapped to. The new global min for the
            dataset as a whole, by default -1
        scaled_max : int, optional
            The value df_global_max is mapped to. The new global max for the
            dataset as a whole, by default 1

        Returns
        -------
        df_scaled: pd.DataFrame
            Scaled version of df.
        """
		scaled_rows = []
		for row in df.itertuples(index=False):
			scaled_row = self._scale_to_range(row,
											  scaled_min,
											  scaled_max,
											  global_min=df_global_min,
											  global_max=df_global_max)
			scaled_rows.append(scaled_row)
		df_scaled = pd.DataFrame(scaled_rows, columns=df_X.columns)
		return df_scaled


	def _scale_seq_to_range(
            self,
            seq,
            scaled_min,
            scaled_max,
            global_min=None,
            global_max=None
            ):
        """Given a sequence of numbers - seq - scale its values to the range
		[scaled_min, scaled_max].

		Default behaviour maps min(seq) to scaled_min and max(seq) to
		scaled_max. To map different values to scaled_min/max, set global_min
        and global_max yourself. Manually controlling the global_min/max
        is useful if you map multiple sequences to the same range but each
        sequence does not contain the same min/max values.

        Parameters
        ----------
        seq : 1D array
            1D array containing numbers.
        scaled_min : int or float
            The minimum value of seq after it has been scaled.
        scaled_max : int or float
            The maximum value of seq after it has been scaled.
        global_min : int or float, optional
            The minimum possible value for elements of seq, by default None.
            If None, this is taken to be min(seq). You will want to set
            global_min manually if you are scaling multiple sequences to the
            same range and they don't all contain global_min.
        global_max : int or float, optional
            The maximum possible value for elements of seq, by default None.
            If None, this is taken to be max(seq). You will want to set
            global_max manually if you are scaling multiple sequences to the
            same range and they don't all contain global_max.

        Returns
        -------
        scaled_seq: 1D np.ndarray
            1D array with all values mapped to the range [scaled_min, scaled_max].
        """
        assert seq.ndim == 1
		assert scaled_min < scaled_max
        assert global_min < global_max

		if global_max is None:
			global_max = np.max(seq)
		if global_min is None:
			global_min = np.min(seq)

		scaled_seq = np.array([self._scale_one_value(value, scaled_min, scaled_max,
										  			 global_min, global_max) \
							   for value in seq])

		return scaled_seq


	def _scale_one_value(
            self,
            value,
            scaled_min,
            scaled_max,
            global_min,
            global_max
            ):
        """Scale value to the range [scaled_min, scaled_max]. The min/max
        of the sequence/population that value comes from are global_min and
        global_max.

        Parameters
        ----------
        value : int or float
            Single number to be scaled
        scaled_min : int or float
            The minimum value that value can be mapped to.
        scaled_max : int or float
            The maximum value that value can be mapped to.
        global_min : int or float
            The minimum value of the population value comes from. Value must
            not be smaller than this.
        global_max : int or float
            The maximum value of the population value comes from. Value must
            not be bigger than this.

        Returns
        -------
        scaled_value: float
            Value mapped to the range [scaled_min, scaled_max] given global_min
            and global_max values.
        """
        assert value >= global_min
        assert value <= global_max
		# Math adapted from this SO answer: https://tinyurl.com/j5rppewr
		numerator = (scaled_max - scaled_min) * (value - global_min)
		denominator = global_max - global_min
        scaled_value = (numerator / denominator) + scaled_min
		return scaled_value


	def _reshape_to_torch_input(
            self,
            df_X,
            df_y,
            batch_first=True,
            y_encoding='label'):
        """Reshapes df_X and df_y into shapes PyTorch accepts. Encodes df_y
        with either label or one-hot encoding if desired. Returns reshaped
        X and y as numpy arrays.

        Parameters
        ----------
        df_X : pd.DataFrame
            DataFrame containing X data in the shape (n_samples, n_features)
        df_y : pd.DataFrame or pd.Series
            DataFrame or Series containing labels.
        batch_first : bool, optional
            Whether to return X in the shape (batch, seq_length, features)
            i.e. with the batch as the first dimention, or with the shape
            (seq_length, batch, features). For easier migration to Keras,
            the default is True (as this is what Keras expects).
        y_encoding : str, optional {'label', 'ohe', None}
            The encoding to apply to df_y. Options are label encoding (0, n-1),
            one-hot encoding or no encoding. The default is 'label' as this is
            what PyTorch expects for multilcass classification problems.

        Returns
        ----------
        X: numpy.ndarray
            Array of values shaped based on batch_first argument.
        y: numpy.ndarray
            Array of labels encoded based on the y_encoding argument.
        """
        # Reshape X values
		X = df_X.values
        if batch_first:
            # Shape is (batch, seq_length, features)
            # This is the default in Keras
            X = X.reshape(-1, self.seq_length, 1)
        else:
            # Shape is (seq_length, batch, features)
            X = X.reshape(self.seq_length, -1, 1)

		# Encode y values
		y_values = df_y.values
        if y_encoding is None:
            y = y_values
        elif y_encoding == 'label':
            label_enc = LabelEncoder()
            y = label_enc.fit_transform(y_values)
        elif y_encoding == 'ohe':
            ohe = OneHotEncoder(sparse=False)
            y = ohe.fit_transform(y_values.reshape(-1, 1))
        else:
            raise ValueError('''Invalid value passed for y_encoding. Please
                            pass 'label' for LabelEncoding, 'ohe' for
                            OneHotEncoding or None for no encoding.''')
		return X, y


	def setup(self, stage):
        """Creates the X/y train/val splits necessary to train and validate
        the model on every GPU used in training.

        First, we load, scale, and reshape the data to be in the correct
        PyTorch format. Then we split it into X/y train/val objects.

        Parameters
        ----------
        stage : str, optional {'fit', 'validate', 'test', None}
            Set this optional argument to use different setup logic depending
            on the stage your model is being used for.

            Note: for this project this argument does nothing.
        """
		# Load, scale, reshape
		df_X_unscaled, df_y = self._load_dfX_dfy()
		df_X_scaled, df_y = self._scale_X_y(df_X_unscaled, df_y,
											scaled_min=self.rescaled_min_val,
											scaled_max=self.rescaled_max_val)

        X, y = self._reshape_to_torch_input(df_X_scaled,
                                            df_y,
                                            batch_first=self.batch_first,
                                            y_encoding=self.y_encoding)
		# Split
		self.X_train, self.X_val, \
			self.y_train, self.y_val = train_test_split(
                                            X, y,
                                            test_size=self.val_split,
											random_state=self.seed,
											shuffle=self.shuffle,
											stratify=y))


	def train_dataloader(self):
		# Create augmented Dataset
		dataset = ElectroAugmenterDataset(
			self.X_train,
			self.y_train,
			self.horizontal_shift,
			self.vertical_shift,
			self.noise_shift,
			self.noise_shift_scale,
			self.multiplier,
			self.seed, # use same seed throughout
			self.aug_pct
		)
		# Create DataLoader
		train_loader = DataLoader(
			dataset,
			self.batch_size,
			self.shuffle,
			num_workers=os.cpu_count()
		)
		return train_loader


	def val_dataloader(self):
		self.X_val = torch.FloatTensor(self.X_val)
		self.y_val = torch.LongTensor(self.y_val)

		dataset = TensorDataset(self.X_val, self.y_val)

		val_loader = DataLoader(
			dataset,
			self.batch_size,
			shuffle=False,
			num_workers=os.cpu_count()
		)
		return val_loader


	def test_dataloader(self):
		raise NotImplementedError('Currently does not support test_dataloader')
