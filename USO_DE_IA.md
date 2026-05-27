O modelo de IA (GOOGLE GEMINI AI) foi a IA que utilizamos para nos auxiliar.
Utilizado um prompt para criar um roteiro para a apresentação:
Prompit: Crie um roteiro para uma apresentação sobre multiplicação de matrizes densas (GEMM) utilizando CPU Sequencial, CUDA Naive e CUDA com Tiling em shared memory. Compare desempenho, tempo de execução e eficiência computacional entre as implementações, incluindo referência com NumPy/cuBLAS. Destaque a importância do paralelismo, da hierarquia de memória da GPU e do uso de tiling para otimização.

Utilizada a IA para fazer correções e melhorias quando necessario nos codigos.
O codigo para fazer o grafico baseado nos resultados das analises foi gerado pela IA:
Prompit: Faça um código para um grafico para apresentar os resultados desses códigos (Foi passado os códigos para a IA saber como esta funcionando e tambem foi passada a linguagem)
metodos = ['CPU Seq', 'NumPy (Ref)', 'CUDA Naive', 'CUDA Tiled']
    tempos = [v_cpu, t_numpy, v_naive, v_tiled]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metodos, tempos, color=['#e74c3c', '#2ecc71', '#3498db', '#9b59b6'])
    plt.ylabel('Tempo de Execução em Segundos (Escala Logarítmica)')
    plt.title(f'Análise de Desempenho GEMM - Matrizes Reais {N}x{N}')
    plt.yscale('log') # Escala logarítmica devido à disparidade de performance
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.6f}s', ha='center', va='bottom', fontweight='bold')
        
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

A IA foi tabmem utilizada para fazer o resumo, a revisão e melhoria do relatorio, tambem adicionando as formatações corretas quando necessario e tambem gerou o código da tabela para o latex a tabela.
Prompt para o resumo: Analise as seções e subsessões deste relatorio do overleaf, sob o templete "SBC" e gere um resumo(Foi passado o relatorio para a IA); inglês e português.
Prompt da tabela: Gere uma tabela chamada "Análise de Desempenho GEMM - Matrizes Reais 512x512". Os dados são: CPU sequencial: 0.292565s, NumPy(Ref): 0.008236s, CUDA Naive (Memória Global): 0.000865s, CUDA Tiled (Shared Memory): 0.000667s. E as marcações do Tempo de Execução em Segundos(Escala Logaritma): 10^-3, 10^-2, 10^-1.
Prompt da revisão: Análise esse relatorio do overleaf e revise as as seções e subseções, de acordo com as normas SBC e adicione algumas formatações onde necessario.

Utilizada a IA para colocar as referencias de forma correta no overleaf.

A inteligência artificial contribui como uma ferramenta no auxilio criação do código, ajudando a implementar e desenvolver, no artigo pedindo para ia não inventar sobre o tema 
Pedimos também baseado no nosso código em C++ utilizando as observação nosso código desenvolver um roteiro do vídeo de forma mais rápida e organizada. Seu maior benefício está na praticidade: ela consegue ajudar nas estruturar tópicos e acelerar tarefas que antes levavam muito tempo.
No caso dos slides, a IA ajuda na organização visual e no resumo das informações. Em artigos, auxilia na pesquisa e na estruturação das ideias.
Por isso, o uso da IA pode ser visto como uma parceria em resumo a foi utiliza mais no suporte técnico.
