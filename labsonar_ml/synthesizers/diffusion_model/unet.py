import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Verificar se os canais de entrada e saída são os mesmos para a conexão residual
        self.same_channels = in_channels == out_channels

        # Flag que indica se a conexão residual deve ou não ser utilizada
        self.is_res = is_res

        # Primeira camada convolucional
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),   # 3x3 kernel | stride 1 | padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # Função de ativação: GELU 
        )

        # Segunda camada convolucional
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),   # 3x3 kernel | stride 1 | padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # Função de ativação: GELU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Se usar conexão residual
        if self.is_res:
            # Aplicando a primeira camada convolucional
            x1 = self.conv1(x)

            # Aplicando a segunda camada convolucional
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            # Se os canais de entrada e saída são os mesmo, adicional conexão residual diretamente
            if self.same_channels:
                out = x + x2
            else:
                # Se não, aplicar uma camada convolucional 1x1 para ajustar as dimensões antes de adicionar a conexão residual
                shortcut = nn.Conv2d(in_channels = x.shape[1], 
                                     out_channels = x2.shape[1], 
                                     kernel_size=1, 
                                     stride=1, 
                                     padding=0).to(x.device)
                
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normaliando saída do tensor
            return out / 1.414 # TODO entender o porque de ser 1.414

        # Se não estiver usando conexão residual, retornar a saída da segunda camada convolucional
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            
            return x2

    # Método para obter o número de canais da saída para esse bloco
    def get_out_channels(self):
        
        return self.conv2[0].out_channels

    # Método para definir o número de canais de saída para esse bloco
    def set_out_channels(self, out_channels):

        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # Cria uma lista de camadas para o bloco de upsampling
        # O bloco consiste de uma camada ConvTranspose2d para upsampling, seguida de duas camadas ResidualConvBlock
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        # Utiliza as camadas para criar um modelo sequential do pytorch
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concate o tensor de entrada x com o tensor de ligação de deslocado/saltado ao longo da dimensão do canal
        x = torch.cat((x, skip), 1)
        
        # Passa o tensor concatenado através do modelo e retorna o output.
        # print(f"forward = {x.size()}")
        x = self.model(x)
        return x

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        # Cria uma lista de camadas para o bloco de downsampling
        # Cada bloco consiste de duas camadas ResidualConvBlock, seguido de uma camada MaxPool2d para downsampling
        self.layers = [ResidualConvBlock(in_channels, out_channels), 
                  ResidualConvBlock(out_channels, out_channels), 
                  nn.MaxPool2d(2)]
        
        # Utiliza as camadas para criar um modelo sequential do pytorch
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # Passa o tensor através do modelo e retorna o output.
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        Esta classe define uma rede neural genérica de uma camada de feed-forward para incorporar dados de entrada de dimensionalidade input_dim num espaço de incorporação de dimensionalidade emb_dim.
        """
        
        self.input_dim = input_dim
        
        # Define as camadas da rede neural
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # Cria um pytorch sequencial model a partir das camadas definidas
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten do tensor de entrada
        x = x.view(-1, self.input_dim)
        # Aplica as camadas do modelo para o tensor flattened

        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 n_feat=256, 
                 n_cfeat=10, # cfeat = features de contexto
                 height=28,
                 batch_size=32):  
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        
        self.in_channels = in_channels  # Número de canais de entrada
        self.n_feat = n_feat            # Número de feature maps intermediários
        self.n_cfeat = n_cfeat          # Número de classes
        self.h = height                 # Assume-se h == w. Deve ser divisível por 4, então 28, 24, 20, 16... # TODO Por quê?
        self.batch_size = batch_size
        # Inicializando a camada convolucional inicial
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Inicializando o caminho downsampling da rede UNet com dois níveis
        self.down1 = UnetDown(n_feat, 2 * n_feat)        # down1: [10, 256, 8, 8]
        self.down2 = UnetDown(2 * n_feat, 4 * n_feat)    # down2: [10, 256, 4,  4]

        # Backup [original]: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((2)), nn.GELU())

        # Embedding do timestep e labels de contexto com uma camada totalmente conectada da rede neural
        # self.timeembed1 = EmbedFC(1, n_feat)
        # self.timeembed2 = EmbedFC(1, n_feat)
        # self.contextembed1 = EmbedFC(n_cfeat, n_feat)
        # self.contextembed2 = EmbedFC(n_cfeat, n_feat)

        # Inicializando o caminho de upsampling da rede UNet com três níveis
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, kernel_size = 2, stride = 2),#(self.batch_size, 4 * n_feat, self.n_feat//4, self.n_feat//4), # upsampling  
            nn.GroupNorm(8, 4 * n_feat), # normalização                       
            nn.ReLU(),
        )
        self.up1 = UnetUp(n_feat * 8, n_feat * 4)
        self.up2 = UnetUp(n_feat * 6, n_feat)

        # Inicializando as últimas camadas convolucionais para mapear o mesmo número de canais da imagem de entrada
        self.out = nn.Sequential(   
            # Reduzir o número de feature maps
            nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1), # in_channels, out_channels, kernel_size, stride:1, padding:0 
            nn.GroupNorm(8, n_feat), # normalização
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map para o mesmo número de canais como entrada
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : imagem de entrada
        t : (batch, n_cfeat)      : timestep
        c : (batch, n_classes)    : label de contexto
        """

        # print(f"initial={x.size()}")
        # Passando a imagem de entrada através da camada convolucional inicial
        x = self.init_conv(x)

        # Passando o resultada para o caminho de downsampling
        down1 = self.down1(x)
        down2 = self.down2(down1)
        
        
        # print(f"init_conv={x.size()}")
        # print(f"down1={down1.size()}")
        # print(f"down2={down2.size()}")

        # Convertendo os feature maps para um vetor e aplicando uma ativação
        
        hiddenvec = self.to_vec(down2)
        # print(f"hiddenvec={hiddenvec.size()}")
        
        # mascarar o contexto se context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # Embedding o contexto e timestep
        # cemb1 = self.contextembed1(c).view(-1, self.n_feat, 1, 1)   # (batch, 2 * n_feat, 1, 1)
        # temb1 = self.timeembed1(t).view(-1, self.n_feat, 1, 1)
        # cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        # temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        ## print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")

        
        up1 = self.up0(hiddenvec)
        # print(f"up0={up1.size()}")

        # Adicionando e multiplicando múltiplos embeddings
        up2 = self.up1(up1, down2) # up2 = self.up1(cemb1 * up1 + temb1, down2)
        # print(f"up1={up2.size()}")
        up3 = self.up2(up2, down1) # up3 = self.up2(cemb2 * up2 + temb2, down1)
        # print(f"up2={up3.size()}")
        x = torch.cat((up3, x), 1)
        # print(f"forward = {x.size()}")
        out = self.out(x)
        # print(f"out={out.size()}")
        return out