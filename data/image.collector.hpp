#ifndef IMAGE_COLLECTOR_HPP
#define IMAGE_COLLECTOR_HPP

#include <string>
#include <vector>
#include <stdexcept> // Para std::runtime_error

/**
 * @brief Clase base abstrata para coletores de dados.abort
 * 
 * Esta classe define a interface comum que todos os coletores específicos
 * devem implementar. Ela não pode ser instanciada diretamente, mas serve
 * como um contrato para classes derivadas.
 */

class ImageCollector {
    public:
    /**
     * @brief Destrutor virtual
     * Essencial para garantir que os destrutores das classes derivadas sejam chamadas
     * corretamente quando um objeto derivado é excluído através de um ponteiro para ImageCollector.
     */
        virtual ~ImageCollector() = default;

    /**
     * @brief Configura o coletor com os parâmetros necessários.
     * Este método deve ser implementado pela classe derivada para processar
     * qualquer configuração específica que o coletor precise.
     * 
     * @param configString Uma string de confiugração (pode ser JSON, XML, caminho de arquivos, etc.).
     * @throw std::runtime_error Se a configuração falhar.
     */
        virtual void configure(const std::string& configString) = 0;

    /**
     * @brief Inicia o professo de coleta de dados.
     * Este é o método principal onde a lógica de coleta da classe derivada será executada.
     * 
     * @return true se a coleta foi iniciada com sucesso (ou concluído),
     *         false caso contrário.
     * @throw std::runtime_error Se ocorrer um erro durante o processo de coleta.
     */
        virtual bool collect() = 0;

    /**
     * @brief Solicita a interrupção do processo de coleta.
     * Útil para coletores que operam de forma assíncrona ou por longos períodos.
     * A implementação deve garantir uma parada graciosa.
     */
        virtual void stopCollection() = 0;

    /**
     * @brief Obtém o status atual do coletor.
     * 
     * @return Uma string descrevendo o status (ex: "ocioso", "coletando", "parado", etc.).
     */
        virtual std::string getStatus() const = 0;

    /**
     * @brief Verifica se o processo de coleta foi concluído
     * 
     * @return true se a coleta for concluída,
     *        false caso contrário.
     */
        virtual bool isCollectionComplete() const = 0;

    protected:
    // Construtor protegido para garantir que a classe só possa ser usada como base.
        ImageCollector() = default;
};

#endif