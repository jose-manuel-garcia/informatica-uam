/*
 *  Copyright (C) 2020 Pablo Castells y Javier Sanz-Cruzado
 *
 *  Este código se ha implementado para la realización de las prácticas de
 *  la asignatura "Búsqueda y minería de información" de 4º del Grado en
 *  Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 *  la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 *  se ciñe a las actividades docentes de dicha asignatura.
 *
 */
package es.uam.eps.bmi.search.index.lucene;

import es.uam.eps.bmi.search.index.AbstractIndexBuilder;
import es.uam.eps.bmi.search.index.Index;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;

/**
 *
 * @author pablo
 */
public class LuceneBuilder extends AbstractIndexBuilder {
    IndexWriter builder;
    protected static FieldType type;
    String indexFolder;

    public LuceneBuilder() {
        type = new FieldType();
        type.setIndexOptions (IndexOptions.DOCS_AND_FREQS);
    }
    
    public void build (String collectionPath, String path) throws IOException {
        indexFolder = path;
        clear(indexFolder);
        IndexWriterConfig iwc = new IndexWriterConfig(new StandardAnalyzer()); 
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        builder = new IndexWriter(FSDirectory.open(Paths.get(indexFolder)), iwc);

        File f = new File(collectionPath);
        if (f.isDirectory()) indexFolder(f);                // A directory containing text files.
        else if (f.getName().endsWith(".zip")) indexZip(f); // A zip file containing compressed text files.
        else indexURLs(f);                                  // A file containing a list of URLs.

        builder.close();
        saveDocNorms(indexFolder);
    }
    
    public void indexText(String text, String path) throws IOException {
        Document doc = new Document();
        Field pathField = new StringField("path", path, Field.Store.YES);
        doc.add(pathField);
        Field field = new Field("content", text, type);
        doc.add(field); 
        builder.addDocument(doc);
    }

    protected Index getCoreIndex() throws IOException {
        return new LuceneIndex(indexFolder);
    }
}
