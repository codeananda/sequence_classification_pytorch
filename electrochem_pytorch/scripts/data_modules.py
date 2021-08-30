import numpy as np
import pytorch_lightning as pl
from pathlib import Path

class UnaugmentedAnalyteDataModule(pl.LightningDataModule):

	def __init__(
            self,
			data_dir='data/',
            batch_size=50,
            seq_length=1002,
            rescaled_min_val=-1,
            rescaled_max_val=1,
            random_seed=42,
            shuffle=True,
            validation_split=0.2,
            ):
		super().__init__()
		self.data_dir = Path(data_dir)
		self.batch_size = batch_size
		self.seq_length = seq_length
		# Scaling params
		self.rescaled_min_val = rescaled_min_val
		self.rescaled_max_val = rescaled_max_val
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

    def _scale_X_y(self, df_X, df_y, scaled_min=-1, scaled_max=1):
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
											  seq_min=df_global_min,
											  seq_max=df_global_max)
			scaled_rows.append(scaled_row)
		df_scaled = pd.DataFrame(scaled_rows, columns=df_X.columns)
		return df_scaled


	def _scale_seq_to_range(
            self,
            seq,
            scaled_min,
            scaled_max,
            seq_min=None,
            seq_max=None):
        """Given a sequence of numbers - seq - scale all of its values to the
		range [scaled_min, scaled_max].

		Default behaviour maps min(seq) to scaled_min and max(seq) to
		scaled_max. To override this, set scaled_min and scaled_max yourself.

        Parameters
        ----------
        seq : array-like
            Array-like structure containing numbers.
        scaled_min : int or float
            The minimum value of seq after it has been scaled.
        scaled_max : int or float
            The maximum value of seq after it has been scaled.
        seq_min : int or float, optional
            The minimum value of the sequence, by default None. If None,
            the minimum value is taken to be min(seq). You may want to set
            seq_min manually if you are scaling multiple sequences to the
            same range and they don't all contain seq_min.
        seq_max : int or float, optional
            The maximum value of the sequence, by default None. If None,
            the maximum value is taken to be max(seq). You may want to set
            seq_max manually if you are scaling multiple sequences to the
            same range and they don't all contain seq_max.

        Returns
        -------
        scaled_seq: np.ndarray
            Array with all values mapped to the range [scaled_min, scaled_max].
        """
		assert scaled_min < scaled_max
		# Default is to use the max of the seq as the min/max
		# Can override this and input custom min and max values
		# if, for example, want to scale to ranges not necesarily included
		# in the data (as in our case with the train and val data)
		if seq_max is None:
			seq_max = np.max(seq)
		if seq_min is None:
			seq_min = np.min(seq)
		assert seq_min < seq_max
		scaled_seq = np.array([self._scale_one_value(value, scaled_min, scaled_max,
										  			 seq_min, seq_max) \
							   for value in seq])

		return scaled_seq


	def _scale_one_value(
            self,
            value,
            scaled_min,
            scaled_max,
            original_min,
            original_max):
		# Scale value into [scaled_min, scaled_max] given the max and min values of the seq
		# it belongs to.
		# Taken from this SO answer: https://tinyurl.com/j5rppewr
		numerator = (scaled_max - scaled_min) * (value - original_min)
		denominator = original_max - original_min
		return (numerator / denominator) + scaled_min


	def _reshape(self, df_X, df_y):
		"""
		Re-shapes df_X and df_y into a format PyTorch LSTMs will accept.
		Namely: (batch, timesteps, features) - just like with Keras.

		Note: you must set batch_first=True in your LSTM layers for this X
			  shape to work.
		"""
		X = df_X.values
		# Correct if batch_first=True in nn.LSTM layers
		X = X.reshape(-1, self.seq_length, 1)
		# PyTorch accepts integer y-values by default
		y_values = df_y.values
		# label_enc = LabelEncoder()
		# y = label_enc.fit_transform(y_values)
		ohe = OneHotEncoder(sparse=False)
		y = ohe.fit_transform(y_values.reshape(-1, 1))
		return X, y


	def _split_X_y(self, X, y):
		X_train, X_val, y_train, y_val = train_test_split(
											X, y,
											test_size=self.val_split,
											random_state=self.seed,
											shuffle=self.shuffle,
											stratify=y)
		return X_train, X_val, y_train, y_val


	def setup(self, stage):
		"""
		Things you want to perform on every GPU.

		It is ok to assign things here.
		"""
		# Load, scale, reshape
		df_X_unscaled, df_y = self._load_X_y_from_columns()
		df_X_scaled, df_y = self._scale_X_y(df_X_unscaled, df_y,
											scaled_min=self.rescaled_min_val,
											scaled_max=self.rescaled_max_val)
		X, y = self._reshape(df_X_scaled, df_y)
		# Split
		self.X_train, self.X_val, \
			self.y_train, self.y_val = self._split_X_y(X, y)


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
