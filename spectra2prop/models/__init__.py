    def forward(self, x):
                x = self.embedding(x)
                        x = x.unsqueeze(1)
                                x = self.transformer_encoder(x)
                                        x = x.mean(dim=1)
                                                x = self.fc(x)
                                                        return xfrom .cnn1d import SpectralCNN, SimpleTransformer
